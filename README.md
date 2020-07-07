# PPRGo

This repository provides the reference implementation of PPRGo for a single machine in TensorFlow 1. PPRGo is a fast GNN able to scale to massive graphs in both single-machine and distributed setups. It was proposed in our paper

**[Scaling Graph Neural Networks with Approximate PageRank](https://www.daml.in.tum.de/pprgo)**   
by Aleksandar Bojchevski, Johannes Klicpera, Bryan Perozzi, Amol Kapoor, Martin Blais, Benedek Rózemberczki, Michal Lukasik, Stephan Günnemann 
Published at ACM SIGKDD 2020.

## Installation
We recommend importing the Anaconda environment saved in `environment.yaml`, which provides the correct TensorFlow and CUDA versions. You can then install the repository using `python setup.py develop`. Note that installing the requirements regularly will most likely result in the wrong CUDA version, since CUDA 10.0 contains a bug that affects PPRGo.

## Run the code
This repository contains a notebook for running training and inference (`run.ipynb`) and a script for running it on a cluster with [SEML](https://github.com/TUM-DAML/seml) (`run_seml.py`).

## Contact
Please contact a.bojchevski@in.tum.de or klicpera@in.tum.de if you have any questions.

## Cite
Please cite our paper if you use the model or this code in your own work:

```
@inproceedings{bojchevski2020pprgo,
  title={Scaling Graph Neural Networks with Approximate PageRank},
  author={Bojchevski, Aleksandar and Klicpera, Johannes and Perozzi, Bryan and Kapoor, Amol and Blais, Martin and R{\'o}zemberczki, Benedek and Lukasik, Michal and G{\"u}nnemann, Stephan},
  booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2020},
  publisher = {ACM},
  address = {New York, NY, USA},
}
```
