# RAlib - Accelrating multireferece alignment

This is the repository that contains gpu-accelerated multireference alignment code for cryo-EM image processing.
The code is used in the 2020 NCHC GPU Hackathon.

Team member:  [Szu Chi Chung](https://github.com/phonchi), [Cheng-Yu Hung](https://github.com/veis5566), [Huei-Lun Siao](https://github.com/oppty1335), [Hung-Yi Wu](https://github.com/Hungyi5), Institute of Statistical Science, Academia Sinica

Mentor: Ryan Jeng, Nvidia


* Multireference alignment (Class averages)
![](https://i.imgur.com/Fhz8VgW.png)

* Reference-free alignment (Average of the first 10 iteration)
![](https://i.imgur.com/4Je3oTt.png)

## Benchmark
* Multireference alignment
We compared the CPU implementation from EMAN2 and use [Ribosome 80s](https://www3.mrc-lmb.cam.ac.uk/relion/index.php?title=Benchmarks_%26_computer_hardware) but downsampling to 90 pixels. The following chart is running on [TWCC](https://www.twcc.ai/) c.super instance. The `xr`, `yr`, `ou`, `maxit` is set to 3,3,36,6 respectively. 

![](https://i.imgur.com/GkXidsN.png)

The speedup is 22x~37x with different reference number.

* Reference-free alignment

The following chart is running on [TWCC](https://www.twcc.ai/) c.super instance. The `ts`, `ou`, `maxit` is set to 1,36,6 respectively. 

![](https://i.imgur.com/mzDF63c.png)


The speedup is 2.4x~9.4x with different 2D shift.

## How to use
### 1. Install 
- Install `EMAN2` and `Sphire`: 
    * Please install [`EMAN2.31`](https://blake.bcm.edu/emanwiki/EMAN2/Install).
    * If you would like to use the version that use CuPy install according to [here](https://github.com/cupy/cupy).
>    Note you may need to relink the nvrtc library `ln -s /usr/local/cuda/lib64/libnvrtc-builtins.so.10.0.130 /usr/local/EMAN2/lib/libnvrtc-builtins.so`

### 2. Setup
- `$ ./install.sh`

### 3. Test
Test data can be downloaded from [here](https://drive.google.com/drive/folders/1BWquinGRMQixtlmjx6edA-LGgzXhldft?usp=sharing).

*  Multi-reference alignment
    - `mpirun -np 4 test_mref_cheng_yu_bdb_cuda.py rib80s_ori_bin.hdf  rib80s_ref.hdf out --ou=36 --xr=3 --yr=3`
    - `mpirun -np 4 test_mref_gpu_align.py rib80s_ori_bin.hdf  rib80s_ref.hdf out --ou=36 --xr=3 --yr=3`

*  Reference-free alignment
    - `mpirun -np 4 test_reffree_gpu_align.py rib80s_ori_bin.hdf  out --ou=36 --xr=3 --yr=3 --ts=1`
