# CPFN: Cascaded Primitive Fitting Networks for High Resolution Point Clouds

This repository contains a PyTorch implementation of the paper:

[CPFN: Cascaded Primitive Fitting Networks for High Resolution Point Clouds](https://arxiv.org/abs/2109.00113). 
<br>
[Eric-Tuan Lê](http://erictuanle.com), 
[Minhyuk Sung](https://mhsung.github.io),
[Duygu Ceylan](http://www.duygu-ceylan.com),
[Radomir Mech](https://research.adobe.com/person/radomir-mech/),
[Tamy Boubekeur](https://perso.telecom-paristech.fr/boubek/),
[Niloy J. Mitra](http://www0.cs.ucl.ac.uk/staff/n.mitra/)
<br>
ICCV 2021


## Introduction

Representing human-made objects as a collection of base primitives has a long history in computer vision and reverse engineering. In the case of high-resolution point cloud scans, the challenge is to be able to detect both large primitives as well as those explaining the detailed parts. While the classical RANSAC approach requires case-specific parameter tuning, state-of-the-art networks are limited by memory consumption of their backbone modules such as PointNet++, and hence fail to detect the fine-scale primitives. We present Cascaded Primitive Fitting Networks (CPFN) that relies on an adaptive patch sampling network to assemble detection results of global and local primitive detection networks. As a key enabler, we present a merging formulation that dynamically aggregates the primitives across global and local scales. Our evaluation demonstrates that CPFN improves the state-of-the-art SPFN performance by 13-14% on high-resolution point cloud datasets and specifically improves the detection of fine-scale primitives by 20-22%

<p align="center">
    <img src="Figures/teaser.png" height=256/>
</p>

## Code
The code will be available soon.

## Cite
Please cite our work if you find it useful:
```latex
@article{cpfnn,
 title={CPFN: Cascaded Primitive Fitting Networks for High Resolution Point Clouds},
 author={Eric-Tuan Lê and Minhyuk Sung and Duygu Ceylan and Radomir Mech and Tamy Boubekeur and Niloy J. Mitra},
 journal={arXiv},
 year={2021}
}
```
