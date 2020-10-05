#!/bin/bash 

set -e
python ./checks/lib_check.py
set -x

# compile CUDA shared lib
nvcc ./cuda/gpu_aln_common.cu ./cuda/gpu_aln_noref.cu -o ./cuda/gpu_aln_pack.so -shared -Xcompiler -fPIC -lcufft -std=c++11

# tell applications.py where to find the CUDA shared lib
sed -i.bkp 's|"..", "..", "cuda"|"cuda"|g' ./bin/applications.py

# tell gpu isac to use the system's python installation
sed -i.bkp "s|/home/schoenf/applications/sphire/v1.1/envs/sphire_1.3/bin|$(dirname $(which sphire))|g" ./bin/sxisac2_gpu.py
