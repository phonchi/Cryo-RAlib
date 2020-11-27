#!/bin/bash 

#*******************************************************************************
#Author (C) 2019, Fabian Schoenfeld (fabian.schoenfeld@mpi-dortmund.mpg.de)
#Copyright (C) 2019, Max Planck Institute of Molecular Physiology, Dortmund

#   This program is free software: you can redistribute it and/or modify it 
#under the terms of the GNU General Public License as published by the Free 
#Software Foundation, either version 3 of the License, or (at your option) any
#later version.

#   This program is distributed in the hope that it will be useful, but WITHOUT
#ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License along with
#this program.  If not, please visit: http://www.gnu.org/licenses/
#*******************************************************************************

set -e
python lib_check.py
set -x

# compile CUDA shared lib
nvcc ./cuda/gpu_aln_common.cu ./cuda/gpu_aln_noref.cu -o ./cuda/gpu_aln_pack.so -shared -Xcompiler -fPIC -lcufft -std=c++11

# tell applications.py where to find the CUDA shared lib
sed -i.bkp 's|"..", "..", "cuda"|"cuda"|g' test_reffree.py

# tell gpu isac to use the system's python installation
#sed -i.bkp "s|/home/schoenf/applications/sphire/v1.1/envs/sphire_1.3/bin|$(dirname $(which sphire))|g" ./bin/sxisac2_gpu.py
