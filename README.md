# Multireferece_alignment

This is the repository that contains gpu-accelerated multireference alignment code for cryo-EM image processing.
The code is used in the 2020 NCHC GPU Hackathon.

Team member:  [Szu Chi Chung](https://github.com/phonchi), [Cheng-Yu Hung](https://github.com/veis5566), [Huei-Lun Siao](https://github.com/oppty1335), [Hung-Yi Wu](https://github.com/Hungyi5), Institute of Science, Academia Sinica

Mentor: Ryan Jeng, Nvidia


## 1. Install 
- Install `EMAN2` and `Sphire`: 
    * Please install [`EMAN2.31`](https://blake.bcm.edu/emanwiki/EMAN2/Install).

## 2. Setup
- `$ ./install.sh`

## 3. Test
- `mpirun -np 4 test_mref_cheng_yu_bdb_cuda.py image_stack.hdf  ref_stack.hdf out --ou=36 --xr=1 --yr=1`