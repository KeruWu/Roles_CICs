# Prominent roles of conditionally invariant components in domain adaptation: theory and algorithms

Code to reproduce the numerical experiments in the [paper](https://arxiv.org/pdf/2309.10301).

### (1) Quick examples of using the package

`./notebook/run_simple.ipynb` is a simple notebook example to run implemented algorithms. 

Package info:

`./src/mdata`: data generation and preprocessing

`./src/mmodel`: pytorch models

`./src/method`: DA algorithms


### (2) Commands to reproduce all experiments in the paper

Each experiment is conducted using 10 distinct seeds. Hyperparameters are chosen via a grid search based on validation accuracy, where the same set of hyperparameters is adopted across seeds. Refer to Appendix D.3 and D.4 for details.

#### 1. SCM, MNIST, CelebA, and DomainNet

The following commands reproduce experiments in Section 5.1.1, 5.2.1, 5.3, and 5.5.
```
# exp = SCM_1, SCM_2, SCM_3, SCM_4 
#       MNIST_rotation_1, MNIST_rotation_2, MNIST_rotation_3, MNIST_rotation_4 
#       CelebA_1, CelebA_2, CelebA_3, CelebA_4, CelebA_5, CelebA_6
#       DomainNet
# seed = 1, 2, ..., 10

python run_exp.py --seeds seed --exp exp
```


#### 2. Camelyon17

The following commands reproduce experiments in Section 5.4.
```
# seed = 1, 2, ..., 10
# algorithm = CIP, 
#             DIP-Pool, JointDIP-Pool                                   (using covariates of unlabeled target data)
#             DIP-Pool_target_labeled, JointDIP-Pool_target_labeled     (using covariates of labeled target data)

python camelyon.py --seeds seed --algorithm algorithm --download False
```


#### 3. Other experiments

The following commands reproduce experiments in Section 5.1.2 and 5.2.2.

```
# DIP on SCM III (Figure 3, 4)
# alpha = 0., .25, .5, .75
python scm3_dip.py --alpha alpha 

# DIP on MNIST III (Figure 3, 6)
# alpha = 0., .25, .5, .75, .9, .95
python mnist3_dip.py --alpha alpha

# JointDIP on SCM III (Figure 3) 
python scm3_jointdip.py 

# JointDIP on MNIST III (Figure 3) 
python mnist3_jointdip.py 
```


#### 4. Download data

All data should be put in the `./data` folder.

- SCM, MNIST: automatically generated / downloaded by `python run_exp.py`.

- CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.
    
    - Extract data from ```img_align_celeba.zip``` to folder ```data/CelebA/img_align_celeba```.

    - Move ```list_attr_celeba.txt``` to folder ```data/CelabA/attr```.

- Camelyon17: https://github.com/p-lambda/wilds.
    
    - `python camelyon.py` with `--download` flag set to True or False. 
    
- DomainNet: https://ai.bu.edu/M3SDA/.

    - Extract data from the zip files (```clipart.zip```, ```infograph.zip```, ```painting.zip```, ```quickdraw.zip```, ```real.zip```, ```sketch.zip```) into their respective folders under data/domainNet/ (```clipart```, ```infograph```, ```painting```, ```quickdraw```, ```real```, ```sketch```).
    
    - Place the text files (```clipart_train.txt```, ```clipart_test.txt```, ```infograph_train.txt```, ```infograph_test.txt```, etc.) in the data/domainNet directory.