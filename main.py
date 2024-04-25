from arguments import commandParser
from slamNet import SlamNet
import torch
from kittiDataset import KittiDataset, KittiDatasetType

from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from itertools import islice
from tqdm import tqdm

# find the huber loss between the estimated pose and the ground truth pose
# The value of delta can be altered
def huber_loss(pose_estimated, actual_pose, delta = 0.1):
    residual = torch.abs(pose_estimated - actual_pose)
    is_small_res = residual < delta
    return torch.where(is_small_res, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))

# RMSE loss
def rmse_loss(pose_estimated, actual_pose):
    residual = torch.abs(pose_estimated - actual_pose)
    return torch.sqrt(torch.mean(residual ** 2))

def lossfunction():
    return huber_loss

def optimizerfunction(net, lr):
    return optim.SGD(net.parameters(), lr=lr)

def set_max_elements(data_len, arg):
    full_data_len = data_len if arg.max_elements == -1 else arg.max_elements
    itermediate = 1 if full_data_len % arg.batch_size > 0 else 0
    batched_data_len = full_data_len//arg.batch_size + itermediate
    return batched_data_len, full_data_len

def validation(model, dataLoader, criterion, validation_data_len, device):
    totalLoss = 0.0
    batched_elements, max_elements = set_max_elements(validation_data_len, arg)
    print(f"number of validation elements: {max_elements}, batched elements: {batched_elements}")
    with torch.no_grad():
        with tqdm(total = batched_elements, desc=f"Validation", position=2) as validationBar:
            for imagePrev, image, pose in islice(dataLoader, 0, batched_elements):
                pose = pose.to(device=device)
                output = model(imagePrev, image)
                loss = criterion(output, pose)
                totalLoss += loss.sum().item()
                validationBar.update(1)

            validationBar.desc = f"Validation loss: {totalLoss / max_elements}"

    return totalLoss

def validationChange(validationLosses, validationLossIdx, epolison=0.01):
    v1 = validationLossIdx
    v2 = (validationLossIdx + 1) % len(validationLosses)
    v3 = (validationLossIdx + 2) % len(validationLosses)
    v4 = (validationLossIdx + 3) % len(validationLosses)
    def difference(valIdx2, valIdx1):
        valLoss2 = validationLosses[valIdx2]
        valLoss1 = validationLosses[valIdx1]
        return (valLoss2 - valLoss1) <= epolison
    if difference(v2, v1) and difference(v3, v2) and difference(v4, v3):
        return True
    return False

# def dummyLoss(x, y, yaw):
#     x_default = torch.randn(3)
#     y_default = torch.randn(3)
#     yaw_default = torch.randn(3)    
    
#     x_mu, x_sigma, x_logvar = x
#     y_mu, y_sigma, y_logvar = y
#     yaw_mu, yaw_sigma, yaw_logvar = yaw

#     x_loss = huber_loss(x_mu, x_default) + huber_loss(x_sigma, x_default) + huber_loss(x_logvar, x_default)
#     y_loss = huber_loss(y_mu, y_default) + huber_loss(y_sigma, y_default) + huber_loss(y_logvar, y_default)
#     yaw_loss = huber_loss(yaw_mu, yaw_default) + huber_loss(yaw_sigma, yaw_default) + huber_loss(yaw_logvar, yaw_default)

#     return x_loss + y_loss + yaw_loss


def train(arg, slamNet, device):

    train_type = KittiDatasetType.eTrain if not arg.dummy_train else KittiDatasetType.eDummyTrain
    train_data = KittiDataset(arg.dataset_path, type=train_type, download=arg.download_dataset, disableExpensiveCheck=True)
    dataLoader = DataLoader(train_data, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers)

    validation_data = KittiDataset(arg.dataset_path, type=KittiDatasetType.eValidation, download=arg.download_dataset, disableExpensiveCheck=True)
    validation_dataLoader = DataLoader(validation_data, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers)

    print(f"Number of training data: {len(train_data)}")
    print(f"Number of validation data: {len(validation_data)}")

    criterion = lossfunction()
    optimizer = optimizerfunction(slamNet, arg.lr)

    decay_step = 0

    # array of args.decay_step elements
    validationLosses = [float('inf')] * arg.decay_step
    validationLossIdx = 0

    batched_elements, max_elements = set_max_elements(len(train_data), arg)
    print(f"Number of train_elements: {max_elements}, batched_elements: {batched_elements}")
    with tqdm(total = arg.epochs, desc=f"Epochs", position=0) as epochBar:
        for epoch in range(arg.epochs):
            epochLoss = 0.0
            runningLoss = 0.0
            with tqdm(total = batched_elements, desc=f"Epoch {epoch} / {arg.epochs}", position=1) as batchBar:
                for i, (imagePrev, image, pose) in enumerate(islice(dataLoader, 0, batched_elements), 0):

                    optimizer.zero_grad()

                    output = slamNet(imagePrev, image)
                    pose = pose.to(device=device)
                    #loss = dummyLoss(x, y, yaw)

                    loss = criterion(output, pose)
                    loss_sum = loss.sum()
                    loss_sum_item = loss_sum.item()
                    loss_sum.backward(retain_graph=False) # NOTE: to avoid RuntimeError: grad can be implicitly created only for scalar outputs

                    optimizer.step()

                    batchBar.update(1)

                    runningLoss += loss_sum_item
                    epochLoss += loss_sum_item
                    if (i+1) % 2000 == 2000:
                        batchBar.desc = f"Epoch {epoch} / {arg.epochs}, loss: {runningLoss / 2000}"
                        runningLoss = 0.0

            epochBar.desc = f"Epoch {epoch} / {arg.epochs}, average training loss: {epochLoss / max_elements}"

            validationLosses[validationLossIdx] = validation(slamNet, validation_dataLoader, criterion, len(validation_data), device)
            validationLossIdx = (validationLossIdx + 1) % arg.decay_step

            if epoch + 1 % arg.decay_step == 0 and validationChange(validationLosses, validationLossIdx, decay_step):
                decay_step += 1
                optimizer = optimizerfunction(slamNet, lr=arg.lr * (arg.decay_rate ** decay_step))

            epochBar.update(1)

    print(f"Finished training, saving model to {arg.save_model}")
    torch.save(slamNet.state_dict(), arg.save_model)

def test(arg, model, model_file, device):
    testData = KittiDataset(arg.dataset_path, type=KittiDatasetType.eTest, download=arg.download_dataset, disableExpensiveCheck=True)
    dataLoader = DataLoader(testData, batch_size=arg.batch_size, shuffle=False, num_workers=arg.num_workers, pin_memory=True)

    model.load_state_dict(torch.load(model_file))

    print(f"Number of test data: {len(testData)}")

    criterion = rmse_loss
    totalLoss = 0.0
    runningLoss = 0.0

    batched_elements, max_elements = set_max_elements(testData, arg)
    print(f"Number of test data: {len(testData)}, batched_elements: {batched_elements}")
    with tqdm(total = batched_elements, desc=f"Testing", position=0) as batchBar:
        with torch.no_grad():
            for i, (imagePrev, image, pose) in enumerate(islice(dataLoader, 0, batched_elements), 0):
                output = model(imagePrev, image)
                pose = pose.to(device=device)
                loss = criterion(output, pose)
                #print(output)
                #print(pose)
                loss_sum = loss.sum()
                loss_sum_item = loss_sum.item()
                totalLoss += loss_sum_item
                runningLoss += loss_sum_item
                batchBar.update(1)
                if (i+1) % 2000 == 2000:
                    batchBar.desc = f"Testing, average running loss: {runningLoss / 2000}"
                    runningLoss = 0.0

    print(f"Finished testing, average loss: {totalLoss / max_elements}")
            

def main(arg):
    expected_shape = (arg.batch_size, 4, 90, 160)
    use_cuda = False if arg.cpu else True
    slamNet = SlamNet(expected_shape, arg.num_particles, is_training=arg.is_training, is_pretrain_obs=arg.is_pretrain_obs, is_pretrain_trans= arg.is_pretrain_trans, use_cuda=use_cuda)
    if arg.cpu:
        slamNet = slamNet.cpu()
        device = torch.device('cpu')
    else:
        slamNet = slamNet.cuda()
        device = torch.device('cuda')

    if arg.test_only:
        test(arg, slamNet, arg.load_model, device)
        return

    train(arg, slamNet, device)
    test(arg, slamNet, arg.save_model, device)


if __name__ == "__main__":
    arg = commandParser()
    main(arg)

