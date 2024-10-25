# Multi-Swap K-means++
This repository contains the code used to run experiments on the multi-swap K-means++ algorithm from https://arxiv.org/pdf/2309.16384.

### Project structure
This project has three files

- multi_swap_kmpp.py contains the bulk of the code. There you can find the implementation of multi-swap k-means++ both with either greedy or vanilla (i.e., brute-force) removal rule
- main.py runs the experiments
- ```utilities.py''' contains utility functions to manage the datasets.

### How to Run
In order to run the experiments you should run
``` main.py dataset_name k local_search_iter lloyd_iter num_experiments out_dir_name'''
where 
- ```dataset_name''' is the name of the dataset
- ```k''' is the target number of centers
- ```local_search_iter''' is the number of local-search iterations
- ```lloyd_iter''' is the number of lloyd iterations executed at the end of the seeding phase
- ```num_experiments''' is the number of independent experiments
- ```out_dir_name''' is the name of the directory where the results are saved.


### Datasets
This code is meant to be used on the datasets KDD-BIO, KDD-PHY and RNA, as described in https://arxiv.org/pdf/2309.16384.
KDD-BIO and KDD-PHY are standard datasets and can be downloaded from https://www.kdd.org/kdd-cup/view/kdd-cup-2004/Data.
RNA is from a bioinformatic dataset (see https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-173#additional-information) and can be downloaded from https://www.openml.org/search?type=data&status=active&id=351.
