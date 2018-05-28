# CPU&GPU version of Semi-global Matching

In this part, the CPU version and GPU version of SGM are implemented. Because I am a freshman on CUDA programming, the implementation of GPU version is only a beta version, which would be improved in the future.

## guassion.h guassion.cpp

These two files implement the gaussian filtering of an image, which would be used on an image before running the SGM algorithm.

## sgm_cpu.cpp

In this file, the cpu version of SGM is implemented by https://github.com/reisub/SemiGlobal-Matching which use the BT algorithm to calculate the pixel cost instead of mutual imformation.

## sgm_gpu.cu

My implementation of GPU version of SGM algorithm. Two parts are designed to run in parallel. One is the computation of pixel cost, the other is the computation of aggregation cost.

## compare.py

This file could be called to valid the correctness of the output of CPU version and GPU version of SGM version.


# ===========================================

To build this project, I'm sorry to recommend you to import the necessary libraries and header files by yourselves, because I am not good at *Cmake*. :-(
