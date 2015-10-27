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
* stage = sleep
* Draw step charts

**Commands**
```
$ python dynamics_graphs.py 300000 0.9999 0.123 sleep --stepchart
```
Graphs are saved in `graphs/`.

**Results**

<img src="https://github.com/ikenox/human-turnover-simulator/wiki/img/3dscat.png" width="600px">

### Plot 3D scatter graph of simulations

#### example

**Commands**

```
$ python sims_to_csv.py   // Do simulations and save results to csv file
$ python csv_to_3dscat.py (csv filepath)
```
The graph are saved in `graphs/`.

**Results**

<img src="https://github.com/ikenox/human-turnover-simulator/wiki/img/fn_and_alpha.png" width="200px">
<img src="https://github.com/ikenox/human-turnover-simulator/wiki/img/interval_angle_dist.png" width="200px">
<img src="https://github.com/ikenox/human-turnover-simulator/wiki/img/interval_ccdf.png" width="200px">
<img src="https://github.com/ikenox/human-turnover-simulator/wiki/img/walking.png" width="600px">
