# influence_function
repo for influence function

cf. [Influence Functions for PyTorch](https://github.com/nimarb/pytorch_influence_functions)


---

# Environments

| item | versions | 
| ---- | -------- | 
| OS | Ubuntu 20.04.5 LTS on WSL2| 
| Python | 3.10.10 on pyenv | 
| Mem | 4.5 GiB available | 


# clone

```
git clone https://github.com/bci-oshita/influence_function.git
cd influence_function
```

# installation

you can choose installation ways bellow `1.` or `2.`


## 1. install from github, directly

```
pip install git+https://github.com/bci-oshita/influence_function.git
```


## 2. install from current directory

```
pip install --upgrade pip
pip install .
```


---

# run

```
make run-cifar10
```

same as `make run`

or

```
make run-mnist
```


# notebook

after run (train/analyze) for mnist/cifar10, you can view the helpful/harmful images in the [notebook](./influ_examples/notebooks/influence_viewer.ipynb)


---
(c) Brains Consulting, inc.
