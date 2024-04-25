import csv
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Generator

from PIL import Image

import itertools

from tqdm import tqdm

import torch
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, calculate_md5
from torchvision.datasets.vision import VisionDataset

import pandas as pd
from enum import Enum

class KittiDatasetType(Enum):
    eTrain = 0
    eTest = 1
    eValidation = 3
    eDummyTrain = 4

class KittiDataset(VisionDataset):

    data_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
    data_raw_url = data_url + "raw_data/"

    filter_scenarios = []

    resources = {
            "data_depth_annotated.zip": "7d1ce32633dc2f43d9d1656a1f875e47",
            "data_depth_velodyne.zip": "20bd6e7dc741520240a0c471392fe9df",
    }


    def __init__(self, root:str, type:KittiDatasetType,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 download: bool = False,
                 remove_finished: bool = False,
                 disableExpensiveCheck: bool = False):
        super().__init__(root, transforms, transform, target_transform)
        self.root = Path(Path(root) / "kitti_dataset")
        self.remove_finished = remove_finished
        self.disableExpensiveCheck = disableExpensiveCheck
        self.shouldDownload = download
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

        if not self._download_folder.exists():
            self._download_folder.mkdir(parents=True)
        if not self._extracted_folder.exists():
            self._extracted_folder.mkdir(parents=True)

        self.scenariosFile = Path(self.root) / "kittiMd5.txt"
        if type == KittiDatasetType.eTrain:
            self.filter_scenarios = ["2011_09_26", "2011_09_28"]
            self.filter_scenarios = ["2011_10_03"]
            self.name = "train"
        elif type == KittiDatasetType.eDummyTrain:
            self.filter_scenarios = ["2011_10_03"]
            self.name = "dummy train"
        elif type == KittiDatasetType.eValidation:
            self.filter_scenarios = ["2011_10_03"]
            self.name = "validation"
        elif type == KittiDatasetType.eTest:
            self.filter_scenarios = ["2011_09_29", "2011_09_30"]
            self.name = "test"
        self.scenarios = self._getScenarios(self.scenariosFile, self.filter_scenarios)

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.datalist = self._parse_datas(self.scenarios)

    def poseGenerator(self, batch_size):
        for index in range(0, len(self.datalist), batch_size):
            poses = []
            for i in range(index, index + batch_size):
                poses.append(self._absPose(i))
            poses = torch.Tensor(poses)
            yield poses

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        imagePrev = self._rgbdPrev(index)
        image = self._rgbd(index)
        pose = self._absPose(index)

        if self.transforms:
            imagePrev = self.transforms(imagePrev)
            image = self.transforms(image)
        return imagePrev, image, pose

    def __len__(self) -> int:
        return len(self.datalist)

    def _parse_datas(self, scenarios) -> List[dict]:
        newDataList = []
        with tqdm(total=len(scenarios), desc=f"reading files for {self.name}") as pbar:
            for folder_file, _ in scenarios:
                datalist = self._parse_folder(folder_file)
                datalistdict = []
                for i in range(0, len(datalist)-1):
                    datalistdict.append({"leftRgbPrev":datalist[i]["leftRgb"],
                                         "leftDepthPrev":datalist[i]["leftDepth"],
                                         **datalist[i+1]})
                newDataList.extend(datalistdict)
                pbar.update(1)
        return newDataList


    def _parse_folder(self, folder_file) -> List[Any]:
        listImages = []
        calib = {}
        for folder_file, _ in self.scenarios:
            if folder_file.endswith("calib.zip"):
                folder_prefix = '_'.join(folder_file.split('_')[:-1])
                calibFile = self._extracted_raw / folder_prefix / "calib_cam_to_cam.txt"
                calib['cam2cam'] = self._calib_to_dict(calibFile)
                calibFile = self._extracted_raw / folder_prefix / "calib_velo_to_cam.txt"
                calib['velo2cam'] = self._calib_to_dict(calibFile)
                calibFile = self._extracted_raw / folder_prefix / "calib_imu_to_velo.txt"
                calib['imu2velo'] = self._calib_to_dict(calibFile)
                continue
            folder_prefix = '_'.join(folder_file.split('_')[:-3])
            folder = folder_file.split('.')[0]
            rgb_folder = self._extracted_raw / folder_prefix / folder / "image_02" / "data"
            depth_folder = self._extracted_depth / "train" / folder / "proj_depth" / "groundtruth" / "image_02"
            if not depth_folder.exists():
                continue
            timestamps = self._timestamps(self._extracted_raw / folder_prefix / folder / "oxts" / "timestamps.txt")
            oxts_keys = self._oxts_keys(self._extracted_raw / folder_prefix / folder / "oxts" / "dataformat.txt")
            oxts_folder = self._extracted_raw / folder_prefix / folder / "oxts" / "data"
            oxt_files = sorted(oxts_folder.iterdir())
            prevOxt = self._extract_oxts(oxts_keys, oxt_files[0])
            prevPose = {'x':0, 'y':0, 'yaw':0}
            posesList = [prevPose]
            for i, file in enumerate(itertools.islice(oxt_files, 1, len(oxt_files))):
                oxt = self._extract_oxts(oxts_keys, file)
                delta = timestamps[i+1] - timestamps[i]
                pose = self._oxts_to_pose(prevPose, prevOxt, oxt, delta)
                posesList.append(pose)
                prevOxt = oxt
            for file in depth_folder.iterdir():
                filename = file.name
                depthImagePath = depth_folder / filename
                rgbImagePath = rgb_folder / filename
                assert int(filename.split('.')[0]) < len(posesList), f"pose index {int(filename.split('.')[0])} < {len(posesList)} out of range"
                pose = posesList[int(filename.split('.')[0])]
                listImages.append({"leftRgb":rgbImagePath, "leftDepth":depthImagePath, "absPose":pose["absolutePose"], "relPose":pose["relativePose"]})
        return listImages

    @property
    def _download_folder(self) -> Path:
        return self.root / "downloaded"

    @property
    def _extracted_folder(self) -> Path:
        return self.root / "extracted"

    @property
    def _extracted_depth(self) -> Path:
        return self._extracted_folder / "depth"

    @property
    def _extracted_raw(self) -> Path:
        return self._extracted_folder / "raw"

    def _oxts_to_pose(self, prevPose, prevOxt, oxt, delta) -> dict:
        # convert oxts to pose
        assert delta > 0
        currentPose = {"x":0, "y":0, "yaw":0}
        # dictionary of 'vf', 'vl', 'vu' and 'ax', 'ay', 'az', 'af', 'al', 'au' 'wx', 'wy', 'wz', 'wf', 'wl', 'wu', 'pos_accuracy', 'vel_accuracy'
        # calculate displacement in x, y and yaw
        x = (oxt['vf']) * delta
        y = (oxt['vl']) * delta
        #yaw = (oxt['vu'] - prevOxt['vu']) * delta
        currentPose['x'] = prevPose['x'] + x
        currentPose['y'] = prevPose['y'] + y
        currentPose['yaw'] = oxt['yaw']
        yaw = currentPose['yaw'] - prevPose['yaw']
        return {"absolutePose":currentPose, "relativePose": {"x":x, "y":y, "yaw":yaw}}

    def _absPose(self, index):
        pose = self.datalist[index]["absPose"]
        x, y, yaw = pose['x'], pose['y'], pose['yaw']
        return torch.Tensor([x, y, yaw])

    def _relPose(self, index):
        pose = self.datalist[index]["relPose"]
        x, y, yaw = pose['x'], pose['y'], pose['yaw']
        return torch.Tensor([x, y, yaw])

    def _rgbdPrev(self, index):
        return self._rgbdTensor(self.datalist[index]["leftRgbPrev"], self.datalist[index]["leftDepthPrev"])

    def _rgbd(self, index):
        return self._rgbdTensor(self.datalist[index]["leftRgb"], self.datalist[index]["leftDepth"])

    def _rgbdTensor(self, rgbFile, depthFile):
        leftRgb = Image.open(rgbFile)
        leftDepth = Image.open(depthFile)

        # convert the image from 3, h, w to 3, 90, 160
        leftRgb = leftRgb.resize((160, 90))
        leftDepth = leftDepth.resize((160, 90))

        leftRgb = transforms.ToTensor()(leftRgb)
        leftDepth = transforms.ToTensor()(leftDepth)
        # combine rgb and depth
        return torch.cat((leftRgb, leftDepth), dim=0)

    def _extract_oxts(self, keys, oxtsFile) -> dict:
        oxts = {}
        with open(oxtsFile, "r") as f:
            line = f.readline()
            values = line.strip().split(" ")
            for i, key in enumerate(keys):
                oxts[key] = float(values[i])
        return oxts

    def _timestamp_generator(self, timestampFile: Path) -> Generator:
        with open(timestampFile, "r") as f:
            for line in f:
                date, time = line.strip().split(" ")
                yield date, time

    def _convertTime(self, time) -> float:
        # convert hh:mm:ss.mmmmmm to seconds
        h, m, s = time.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    def _timestamps(self, timestampFile: Path) -> dict:
        timestamps = {}
        gen = self._timestamp_generator(timestampFile)
        _, firstTimeStamp = next(gen)
        firstTime = self._convertTime(firstTimeStamp)
        timestamps[0] = 0
        for i, (date, time) in enumerate(gen):
            timestamps[i+1] = self._convertTime(time) - firstTime
            assert timestamps[i+1] - timestamps[i] > 0

        return timestamps

    def _oxts_keys(self, oxtsFile: Path) -> List:
        oxts_keys = []
        with open(oxtsFile, "r") as f:
            for line in f:
                key = line.strip().split(" ")[0]
                key = key[:-1] # remove ":" at the end
                oxts_keys.append(key)
        return oxts_keys

    def _calib_to_dict(self, calibFile: Path) -> dict:
        calib = {}
        with open(calibFile, "r") as f:
            for line in f:
                key, *values = line.strip().split(" ")
                key = key[:-1] # remove ":" at the end
                if key == 'calib_time':
                    continue
                calib[key] = [float(value) for value in values]
        return calib


    def _getScenarios(self, scenariosFile, scenariosFilter) -> List[Tuple[str, str]]:
        if not scenariosFile.exists():
            assert False
        with open(scenariosFile, "r") as f:
            file, md5 = zip(*[line.strip().split(" ") for line in f])
            if scenariosFilter is None or len(scenariosFilter) == 0:
                return list(zip(file, md5))
            filterFiles = [f for f in file if any([f.startswith(s) for s in scenariosFilter])]
            file, md5 = zip(*[(f, m) for f, m in zip(file, md5) if f in filterFiles])
        return list(zip(file, md5))

    def _check_exists(self) -> bool:
        if not self._extracted_folder.exists():
            print(f"{self._extracted_folder} doesn't exist")
            return False
        if not self._extracted_depth.exists():
            print(f"{self._extracted_depth} doesn't exist")
            return False
        if not self._extracted_raw.exists():
            print(f"{self._extracted_raw} doesn't exist")
            return False

        if not (self._extracted_depth).exists():
            print(f"{self._extracted_depth} doesn't exist")
            return False
        for file, _ in self.scenarios:
            if file[-9:] != "calib.zip":
                folder_prefix = '_'.join(file.split('_')[:-3])
                folder = file.split('.')[0]
                if not (self._extracted_raw / folder_prefix / folder).exists():
                    print(f"{self._extracted_raw / folder_prefix / folder} doesn't exist")
                    return False
            else:
                folder_prefix = '_'.join(file.split('_')[:-1])
                expected_files = ['calib_cam_to_cam.txt', 'calib_imu_to_velo.txt', 'calib_velo_to_cam.txt']
                for expected_file in expected_files:
                    if not (self._extracted_raw / folder_prefix / expected_file).exists():
                        print(f"{self._extracted_raw / folder_prefix / expected_file} doesn't exist")
                        return False

        def expensiveCheck(dictFileMd5, folder):
            download_folder = self._download_folder;
            for file, md5 in dictFileMd5:
                if not check_integrity(str(download_folder / file), md5):
                    print("{} doesn't have md5 {}".format(file, md5))
                    return False
            return True

        if self.shouldDownload:
            return expensiveCheck(self.resources.items(), self._extracted_depth) and expensiveCheck(self.scenarios, self._extracted_raw)
        if not self.disableExpensiveCheck:
            return expensiveCheck(self.resources.items(), self._extracted_depth) and expensiveCheck(self.scenarios, self._extracted_raw)
        return True

    def _generate_url(self, file) -> str:
        raw_suffix = ["_calib.zip", "_sync.zip", "_tracklets.zip", "_extract.zip"]
        if any(suffix in file for suffix in raw_suffix):
            if file.endswith("_calib.zip"):
                return f"{self.data_raw_url}{file}"
            prefixFile = "_".join(file.split("_")[:-1])
            return f"{self.data_raw_url}{prefixFile}/{file}"
        return f"{self.data_url}{file}"

    def download(self) -> None:
        if self._check_exists():
           return
        for file, md5 in self.scenarios:
            url = self._generate_url(file)
            download_folder = str(self._download_folder)
            extract_folder = str(self._extracted_raw)
            print(f"Downloading {url} to {download_folder} and extracting to {extract_folder}")
            download_and_extract_archive(url, download_root=download_folder, extract_root=extract_folder, filename=file, md5=md5, remove_finished=self.remove_finished)
        for file, md5 in self.resources.items():
            url = self._generate_url(file)
            download_folder = str(self._download_folder)
            extract_folder = str(self._extracted_depth)
            download_and_extract_archive(url, download_root=download_folder, extract_root=extract_folder, filename=file, md5=md5, remove_finished=self.remove_finished)

