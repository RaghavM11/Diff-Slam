# Diff-Slam-Extension to NuScenes 
#deeprob project work

##Differentiable SLAM-net Architecture

![Image Alt text](arch/Diff-arch.png    "Differentiable SLAM-net Architecture")


# Installation [Under Progress]

## Cloning The Repo

To clone the entirety of the repository:
```
git clone git@github.com:RaghavM11/Diff-Slam.git
git submodule update --init --recursive
```

## Installation via `pipenv`

There's multiple ways to install this correctly, but Dylan likes `pipenv` and it seems to work okay for him.

Tested on:
- Ubuntu 22.04
- CMake version 3.17 (>= recommended)
-

Install requriments using : 

```

pip install -r /path/to/requirements.txt

```
## Installation via Anaconda

1. Setup Conda env. (should have miniconda installed)
````
conda create -n myenv python=3.9 
````
2. Activate conda 

```
conda activate myenv

```
3. Install dependecies in conda env following Via Installation Script 



## Contact
If you have any questions or suggestions about this repo, please feel free to contact (imraghav@umich.edu).

## Citation
If you find part of this work useful please cite applications and please consider giving this repo a star

For citing original work by authors refer the following BibTeX entry.

```BIBTEX

@article{DBLP:journals/corr/abs-2105-07593,
  author       = {P{\'{e}}ter Karkus and
                  Shaojun Cai and
                  David Hsu},
  title        = {Differentiable SLAM-net: Learning Particle {SLAM} for Visual Navigation},
  journal      = {CoRR},
  volume       = {abs/2105.07593},
  year         = {2021},
  url          = {https://arxiv.org/abs/2105.07593},
  eprinttype    = {arXiv},
  eprint       = {2105.07593},
  timestamp    = {Tue, 18 May 2021 18:46:40 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2105-07593.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

```

## License
All code in this repository is under the [MIT License].

## Acknowledgement
Diff Slam is based on the following work by Peter Karkus Shaojun Cai and David Hsu
https://sites.google.com/view/slamnet
