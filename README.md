Unsupervised Adversarially-Robust Representation Learning on Graphs
===============================================================================

About
-----

This project is the implementation of the paper "Unsupervised Adversarially-Robust Representation Learning on Graphs".

This repo contains the codes, data and results reported in the paper.

Dependencies
-----

The script has been tested running under Python 3.7.7, with the following packages installed (along with their dependencies):

- `numpy==1.18.1`
- `scipy==1.4.1`
- `scikit-learn==0.23.1`
- `gensim==3.8.0`
- `networkx==2.3`
- `tqdm==4.46.1`
- `torch==1.4.1`
- `torch_geometric==1.5.0`
  - torch-spline-conv==1.2.0
  - torch-scatter==2.0.4
  - torch-sparse==0.6.0

Some Python module dependencies are listed in `requirements.txt`, which can be easily installed with pip:

```
pip install -r requirements.txt
```

In addition, CUDA 10.0 has been used in our project. Although not all dependencies are mentioned in the installation instruction links above, you can find most of the libraries in the package repository of a regular Linux distribution.


Usage: Model Training
-----
Given the adjacency matrix and node attribute matrix of input graph, our model aims to learn a robust graph representation.

Some example data formats are given in ```data``` folder.

When using your own dataset, you must provide:

* an N by N adjacency matrix (N is the number of nodes).

### Main Script
The help information of the main script ```train.py``` is listed as follows:

    python train.py -h
    
    usage: train.py [-h][--dataset] [--pert-rate] [--threshold] [--save-dir]
    
    optional arguments:
      -h, --help                Show this help message and exit
      --dataset                 str, The dataset to be perturbed on [cora, citeseer, polblogs].
      --rate                    float, Perturbation budget of graph topology.
      --eps-x                   float, Perturbation budget of node attributes.
      --tau                     float, The soft margin of robust hinge loss.
      
### Demo
We include all three benchmark datasets Cora, Citeseer and Polblogs in the ```data``` directory.
Then a demo script is available by calling ```train.py```, as the following:

    python train.py --data-name cora --rate 0.4 --eps-x 0.1 --tau 0.005
      
### Evaluations
We provide the evaluation codes on the node classification task here. 
We evaluate on three real-world datasets Cora, Citeseer and Polblogs. 

#### Evaluation Script
The help information of the main script ```eval.py``` is listed as follows:
The help information of the evaluation script is listed as follows:

    python . -h
    
    usage: . [-h][--dataset] [--pert-rate] [--dimensions] [--load-dir]
    
    optional arguments:
      -h, --help                Show this help message and exit
      --dataset                 str, The dataset to be evluated on [cora, citeseer, polblogs].
      --rate                    float, Perturbation budget of graph topology.
      --eps-x                   float, Perturbation budget of node attributes.


