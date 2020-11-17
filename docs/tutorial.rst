********
Tutorial
********


* 
  Multi-reference alignment


  * ``mpirun -np 4 test_mref_cheng_yu_bdb_cuda.py rib80s_ori_bin.hdf  rib80s_ref.hdf out --ou=36 --xr=3 --yr=3``
  * ``mpirun -np 4 test_mref_gpu_align.py rib80s_ori_bin.hdf  rib80s_ref.hdf out --ou=36 --xr=3 --yr=3``

* 
  Reference-free alignment


  * ``mpirun -np 4 test_reffree_gpu_align.py rib80s_ori_bin.hdf  out --ou=36 --xr=3 --yr=3 --ts=1``
