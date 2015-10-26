# Simulation of human turnover model

This program simulates human turnover model, and generate graphs of its results.

## Reference

Simulation model of this program is implementation of human turnover model which is proposed in following research paper.

> [Human turnover dynamics during sleep: statistical behavior and its modeling.](http://www.ncbi.nlm.nih.gov/pubmed/24730888)
> Phys Rev E Stat Nonlin Soft Matter Phys. 2014 Mar;89(3):032721. Epub 2014 Mar 31.
> Yoneyama M, Okuma Y, Utsumi H, Terashi H, Mitoma H.

#### Notice

Since details of this simulation model are constructed with guesswork from origin research papers, this simulation model may differ with origin models.

## Required

* python 2.7
* pip

## Getting started

```
$ cd repository_root
$ pip install -r requirements.txt
```

## How to use

### Plot graph of a simulation

#### example

* Number of steps = 300000
* Probability p = 0.9999
* K = 0.123
* stage = wake
* Draw step charts

**Command**
```
python dynamics_graphs.py 300000 0.9999 0.123 wake --stepchart
```

**Results**



### Plot 3D scatter graph of simulations

example```

```