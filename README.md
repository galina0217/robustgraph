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
- `tqdm==4.46.1`
- `torch==1.4.1`

Some Python module dependencies are listed in `requirements.txt`, which can be easily installed with pip:

```
pip install -r requirements.txt
```

In addition, CUDA 10.0 has been used in our project. Although not all dependencies are mentioned in the installation instruction links above, you can find most of the libraries in the package repository of a regular Linux distribution.


Usage: Model Training
-----
Given the adjacency matrix and node attribute matrix of input graph, our model aims to learn a robust graph representation.

We include all three benchmark datasets Cora, Citeseer and Polblogs in the ```data``` directory.

When using your own dataset, you must provide:

* an N by N adjacency matrix (N is the number of nodes).

### Main Script
The help information of the main script ```train.py``` is listed as follows:

    python train.py -h
    
    usage: train.py [-h][--dataset] [--pert-rate] [--threshold] [--save-dir]
    
    optional arguments:
      -h, --help                Show this help message and exit
      --dataset                 str, The dataset to be perturbed on [cora, citeseer, polblogs].
      --alpha                   float, Perturbation budget of graph topology.
      --epsilon                 float, Perturbation budget of node attributes.
      --tau                     float, The soft margin of robust hinge loss.
      --critic                  str, Critic function ('inner product', 'bilinear' or 'separable').
      --hinge                   bool, Whether to use robust hinge loss.
      --dim                     int, The output dimension of GNN.
      --gpu                     str, which gpu to use.
      --save-model              bool, Whether to save the learned model.
      --show-task               bool, Whether to exhibit the results of downstream task during training.
      --show-attack             bool, Whether to exhibit the attack process during training.
      
### Demo
Then a demo script is available by calling ```train.py```, as the following:

    python train.py --data-name cora --alpha 0.4 --epsilon 0.1 --tau 0.005
      

Usage: Evaluation
-----
We provide the evaluation codes on the node classification task here. 
We evaluate on three real-world datasets Cora, Citeseer and Polblogs. 

### Evaluation Script
The help information of the main script ```eval.py``` is listed as follows:
The help information of the evaluation script is listed as follows:

    python . -h
    
    usage: . [-h][--dataset] [--pert-rate] [--dimensions] [--load-dir]
    
    optional arguments:
      -h, --help                Show this help message and exit
      --dataset                 str, The dataset to be evluated on [cora, citeseer, polblogs].
      --alpha                   float, Perturbation budget of graph topology.
      --epsilon                 float, Perturbation budget of node attributes.
      --model                   str, The model to load.
      --critic                  str, The critic function of the loaded model.
      --hinge                   bool, Whether to use robust hinge loss.
      --dim                     int, The output dimension of the loaded model.
      --gpu                     str, which gpu to use.
      
### Demo
Then a demo script is available by calling ```eval.py```, as the following:

    python eval.py --data-name cora --alpha 0.2 --epsilon 0.1 --model model.pkl
      
