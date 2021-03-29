# Cryo-RAlib - Accelrating alignment in cryo-EM

This is the repository that contains GPU-accelerated multireference alignment code for cryo-EM image processing.
The code is used in the 2020 NCHC GPU Hackathon.

Team member:  [Szu Chi Chung](https://github.com/phonchi), [Cheng-Yu Hung](https://github.com/veis5566), [Huei-Lun Siao](https://github.com/oppty1335), [Hung-Yi Wu](https://github.com/Hungyi5), Institute of Statistical Science, Academia Sinica

Mentor: Ryan Jeng, Nvidia


* Multireference alignment (Class averages)
![](https://i.imgur.com/Fhz8VgW.png)

* Reference-free alignment (Average of the first 10 iterations)
![](https://i.imgur.com/4Je3oTt.png)

## Manuscript:

Cryo-RALib -- a modular library for accelerating alignment in cryo-EM. Szu-Chi Chung, Cheng-Yu Hung, Huei-Lun Siao, Hung-Yi Wu, Wei-Hau Chang, I-Ping Tu. https://arxiv.org/abs/2011.05755



## Benchmark
* Multireference alignment
We compared the CPU implementation from EMAN2 and used [Ribosome 80s benchmark dataset](https://www3.mrc-lmb.cam.ac.uk/relion/index.php?title=Benchmarks_%26_computer_hardware) but downsampling to 90 pixels. The following chart is running on [TWCC](https://www.twcc.ai/) c.super instance. The `xr`, `yr`, `ou`, `maxit` is set to 3,3,36,6 respectively. 

![](https://i.imgur.com/GkXidsN.png)

The speedup is 22x~37x with different reference number.

* Reference-free alignment

The following chart is running on [TWCC](https://www.twcc.ai/) c.super instance. The `ts`, `ou`, `maxit` is set to 1,36,6 respectively. 

![](https://i.imgur.com/mzDF63c.png)


The speedup is 2.4x~9.4x with different 2D shifts.

## How to use the library
### 1. Install 
- Install `EMAN2` and `Sphire`: 
    * Please install [`EMAN2.31`](https://blake.bcm.edu/emanwiki/EMAN2/Install).
    * If you would like to use the version that uses CuPy, please install according to [here](https://github.com/cupy/cupy).
>    Note you may need to relink the nvrtc library `ln -s /usr/local/cuda/lib64/libnvrtc-builtins.so.10.0.130 /usr/local/EMAN2/lib/libnvrtc-builtins.so`

### 2. Setup
- `$ ./install.sh`
- You may need to modify the shebang line in the python scripts to point out the location of the EMAN2 library. 

### 3. Test
Test data can be downloaded from [here](https://drive.google.com/drive/folders/1BWquinGRMQixtlmjx6edA-LGgzXhldft?usp=sharing). See the [Example Notebook](notebook/00_Multireference_Alignment.ipynb) for more details.

*  Multi-reference alignment
    - `mpirun -np 4 test_mref_cheng_yu_bdb_cuda.py rib80s_ori_bin.hdf  rib80s_ref.hdf out --ou=36 --xr=3 --yr=3`
    - `mpirun -np 4 test_mref_gpu_align.py rib80s_ori_bin.hdf  rib80s_ref.hdf out --ou=36 --xr=3 --yr=3`

*  Reference-free alignment
    - `mpirun -np 4 test_reffree_gpu_align.py rib80s_ori_bin.hdf  out --ou=36 --xr=3 --yr=3 --ts=1`

## Notebook
The python environment exposed by EMAN2 can be couple with CuPy and other libraries for drop-in acceleration and visualization. See the [Example Notebook](notebook/02_CuPy_Image_Processing_rot_shift2d.ipynb) where we accelerate the rotation and shift operations by five-fold and visualize the results.

The library can be used for exploratory data analysis as demonstrated in the notebook. You will need the [utils_ralib.py](src/utils_ralib.py) and [scikit-learn](https://scikit-learn.org/stable/) for analysis.

## License
Cryo-RAlib is open source software released under the [GNU General Public License, Version 3](https://github.com/phonchi/Cryo-RAlib/blob/master/LICENSE).

## Credit
Some of the code is built upon Cuda code from [gpu_isac 2.3.2](http://sphire.mpg.de/wiki/doku.php?id=gpu_isac).
