********************************************
Cryo-RAlib: Accelrating alignment in cryo-EM
********************************************

Goals
=====
Multireference alignment is a common algorithm in cryo-EM image processing.
Cryo-RAlib provides a gpu-accelerated version of multireference alignment based on the ``EMAN2`` package.
 



* 
  Multireference alignment (Class averages)

  .. image:: https://i.imgur.com/Fhz8VgW.png
     :target: https://i.imgur.com/Fhz8VgW.png
     :alt: 


* 
  Reference-free alignment (Average of the first 10 iterations)

  .. image:: https://i.imgur.com/4Je3oTt.png
     :target: https://i.imgur.com/4Je3oTt.png
     :alt: 


Benchmark
---------


* Multireference alignment
  We compared the CPU implementation from EMAN2 and used `Ribosome 80s benchmark dataset <https://www3.mrc-lmb.cam.ac.uk/relion/index.php?title=Benchmarks_%26_computer_hardware>`_ but downsampling to 90 pixels. The following chart is running on `TWCC <https://www.twcc.ai/>`_ c.super instance. The ``xr``\ , ``yr``\ , ``ou``\ , ``maxit`` is set to 3,3,36,6 respectively. 


.. image:: https://i.imgur.com/GkXidsN.png
   :target: https://i.imgur.com/GkXidsN.png
   :alt: 


The speedup is 22x~37x with different reference number.


* Reference-free alignment

The following chart is running on `TWCC <https://www.twcc.ai/>`_ c.super instance. The ``ts``\ , ``ou``\ , ``maxit`` is set to 1,36,6 respectively. 


.. image:: https://i.imgur.com/mzDF63c.png
   :target: https://i.imgur.com/mzDF63c.png
   :alt: 


The speedup is 2.4x~9.4x with different 2D shift.




Notebook
--------

The python environment exposed by EMAN2 can be couple with CuPy and other libraries for drop-in acceleration and visualization. See the `Example Notebook <notebook/02_CuPy_Image_Processing_rot_shift2d.ipynb>`_ where we accelerate the rotation and shift operations by five-fold and visualize the results.

The library can be used for exploratory data analysis as demonstrated in the notebook. You will need the `utils_ralib.py <src/utils_ralib.py>`_ and `scikit-learn <https://scikit-learn.org/stable/>`_ for analysis.


2020 NCHC GPU Hackathon
=======================
The code is used in the 2020 NCHC GPU Hackathon.

Team member:  `Szu Chi Chung <https://github.com/phonchi>`_\ , `Cheng-Yu Hung <https://github.com/veis5566>`_\ , `Huei-Lun Siao <https://github.com/oppty1335>`_\ , `Hung-Yi Wu <https://github.com/Hungyi5>`_\ , Institute of Statistical Science, Academia Sinica

Mentor: Ryan Jeng, Nvidia


Credit
------

Some of the code is built upon Cuda code from `gpu_isac 2.3.2 <http://sphire.mpg.de/wiki/doku.php?id=gpu_isac>`_.
