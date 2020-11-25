#!/usr/local/EMAN2/bin/python
from __future__ import print_function
#******************************************************************************
#* 
#* GPU based reference free alignment
#* 
#* Author (C) Szu-Chi Chung 2020 (steve2003121@gmail.com)
#*            Cheng-Yu Hung 2020 (veisteak@gmail.com)
#*            Huei-Lun Siao 2020 (oppty1335@gmail.com)
#*            Hung-Yi Wu 2020 (say66969@gmail.com)
#*            Markus Stabrin 2019 (markus.stabrin@mpi-dortmund.mpg.de)
#*            Fabian Schoenfeld 2019 (fabian.schoenfeld@mpi-dortmund.mpg.de)
#*            Thorsten Wagner 2019 (thorsten.wagner@mpi-dortmund.mpg.de)
#*            Tapu Shaikh 2019 (tapu.shaikh@mpi-dortmund.mpg.de)
#*            Adnan Ali 2019 (adnan.ali@mpi-dortmund.mpg.de)
#*            Luca Lusnig 2019 (luca.lusnig@mpi-dortmund.mpg.de)
#*            Toshio Moriya 2019 (toshio.moriya@kek.jp)
#*            Pawel A.Penczek, 09/09/2006 (Pawel.A.Penczek@uth.tmc.edu)
#*
#*  Copyright (C) 2020 SABID Laboratory, Institute of Statistical Science, Academia Sinica
#*  Copyright (c) 2019 Max Planck Institute of Molecular Physiology
#*  Copyright (c) 2000-2006 The University of Texas - Houston Medical School
#* 
#*    This program is free software: you can redistribute it and/or modify it 
#* under the terms of the GNU General Public License as published by the Free 
#* Software Foundation, either version 3 of the License, or (at your option) any
#* later version.
#* 
#*    This program is distributed in the hope that it will be useful, but WITHOUT
#* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#* FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#* 
#*    You should have received a copy of the GNU General Public License along with
#* this program.  If not, please visit: http://www.gnu.org/licenses/
#* 
#******************************************************************************/



import os
import global_def
from   global_def     import *
from   user_functions import *
from   optparse       import OptionParser
import sys
import sparx as spx

###############################################################
########################################################### GPU

import math
import numpy as np
import ctypes
from EMAN2 import EMNumPy

CUDA_PATH = os.path.join(os.path.dirname(__file__),  "..", "cuda", "")
cu_module = ctypes.CDLL( CUDA_PATH + "gpu_aln_pack.so" )
float_ptr = ctypes.POINTER(ctypes.c_float)

cu_module.ref_free_alignment_2D_init.restype = ctypes.c_ulonglong
cu_module.pre_align_init.restype = ctypes.c_ulonglong

class Freezeable( object ):

    def freeze( self ):
        self._frozen = None

    def __setattr__( self, name, value ):
        if hasattr( self, '_frozen' )and not hasattr( self, name ):
            raise AttributeError( "Error! Trying to add new attribute '%s' to frozen class '%s'!" % (name,self.__class__.__name__) )
        object.__setattr__( self, name, value )

class AlignConfig( ctypes.Structure, Freezeable ):
    _fields_ = [ # data param
                 ("sbj_num", ctypes.c_uint),            # number of subject images we want to align
                 ("ref_num", ctypes.c_uint),            # number of reference images we want to align the subjects to
                 ("img_dim", ctypes.c_uint),            # image dimension (in both x- and y-direction)
                 # polar sampling parameters
                 ("ring_num", ctypes.c_uint),           # number of rings when converting images to polar coordinates
                 ("ring_len", ctypes.c_uint),           # number of rings when converting images to polar coordinates
                 # shift parameters
                 ("shift_step",  ctypes.c_float),        # step range when applying translational shifts
                 ("shift_rng_x", ctypes.c_float),        # translational shift range in x-direction
                 ("shift_rng_y", ctypes.c_float) ]       # translational shift range in y-direction

class AlignParam( ctypes.Structure, Freezeable ):
    _fields_ = [ ("sbj_id",  ctypes.c_int),
                 ("ref_id",  ctypes.c_int),
                 ("shift_x", ctypes.c_float),
                 ("shift_y", ctypes.c_float),
                 ("angle",   ctypes.c_float),
                 ("mirror",  ctypes.c_bool) ]
    def __str__(self):
            return "s_%d/r_%d::(%d,%d;%.2f)" % (self.sbj_id, self.ref_id, self.shift_x, self.shift_y, self.angle) \
                    +("[M]" if self.mirror else "")

aln_param_ptr = ctypes.POINTER(AlignParam)

def get_c_ptr_array( emdata_list ):
    ptr_list = []
    for img in emdata_list:
        img_np = EMNumPy.em2numpy( img )
        assert img_np.flags['C_CONTIGUOUS'] == True
        assert img_np.dtype == np.float32
        img_ptr = img_np.ctypes.data_as(float_ptr)
        ptr_list.append(img_ptr)
    return (float_ptr*len(emdata_list))(*ptr_list)

def print_gpu_info( cuda_device_id ):
    cu_module.print_gpu_info( ctypes.c_int(cuda_device_id) )

########################################################### GPU
###############################################################

def ali2d_base_gpu_isac_CLEAN(
    stack, outdir, maskfile=None, ir=1, ou=-1, rs=1, xr="4 2 1 1", yr="-1", ts="2 1 0.5 0.25",
    nomirror = False, dst=0.0, center=-1, maxit=0, CTF=False, snr=1.0,
    Fourvar=False, user_func_name="ref_ali2d", random_method = "", log = None,
    number_of_proc = 1, myid = 0, main_node = 0, mpi_comm = None, write_headers = False,
    mpi_gpu_proc=False, gpu_class_limit=0, cuda_device_occ=0.9):

    """
    ISAC hardcoded defaults:
        maskfile = None
        ir       = 1     # inner radius
        rs       = 1     # ring step
        dst      = 90.0
        maxit    = 14
        snr      = 1.0
        Fourvar  = False
        user_func_name = "ref_ali2d"
        random_method  = ""
        write_headers  = False

    PROFILE-GPU    Toxin
        alignment    52s
        (..)         0s
        resampling    56s     <-- resampling an issue but already spread across all mpi_procs
        sxheader     2s

    PROFILE-CPU
        alignment    169s
        (..)         0
        resampling    57      <--
        sxheader     2
    """
    #----------------------------------[ local imports ]

    import os
    import sys
    import mpi
    import math
    import time
    import alignment
    import statistics
    import fundamentals
    import utilities as util

    from filter import filt_ctf
    from pixel_error import pixel_error_2D

    import user_functions
    from sp_applications   import MPI_start_end
    user_func = user_functions.factory[user_func_name]

    # sanity check: gpu procs sound off
    if not mpi_gpu_proc: return []

    #----------------------------------[ setup ]
    
    print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "ali2d_base_gpu_isac_CLEAN() :: MPI proc["+str(myid)+"] run pre-alignment CPU setup" )

    # mpi communicator sanity check
    assert mpi_comm != None

    # set alignment search ranges 
    yrng = xrng = util.get_input_from_string(xr)
    step = util.get_input_from_string(ts)
    
    # polar conversion (ring) parameters
    first_ring = int(ir)
    last_ring  = int(ou)
    rstep      = int(rs)

    # alignment iterations
    if int(maxit)==0:
        max_iter  = 10
        auto_stop = True
    else:
        max_iter  = int(maxit)
        auto_stop = False

    # determine global index values of particles handled by this process
    data = stack
    total_nima = len(data)
    total_nima = mpi.mpi_reduce(total_nima, 1, mpi.MPI_INT, mpi.MPI_SUM, 0, mpi_comm)
    total_nima = mpi.mpi_bcast (total_nima, 1, mpi.MPI_INT, main_node, mpi_comm)[0]

    list_of_particles = list(range(total_nima))
    image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
    list_of_particles = list_of_particles[image_start:image_end] # list of global indices

    nima = len(list_of_particles)
    assert( nima == len(data) ) # sanity check

    # read nx and broadcast to all nodes
    # NOTE: nx is image size and images are assumed to be square
    if myid == main_node:
        nx = data[0].get_xsize()
    else:
        nx = 0
    nx = util.bcast_number_to_all(nx, source_node=main_node, mpi_comm=mpi_comm)

    if CTF:
        phase_flip = True
    else:
        phase_flip = False

    CTF = False # okay..?

    # set default value for the last ring if none given
    if last_ring == -1: last_ring = nx/2-2

    # sanity check for last_ring value
    if last_ring + max([max(xrng), max(yrng)]) > (nx-1) // 2:
        ERROR( "Shift or radius is too large - particle crosses image boundary", "ali2d_MPI", 1 )

    # mask (note: ISAC hardcodes default circular mask)
    mask = util.model_circle(last_ring, nx, nx)

    # image center
    cny = cnx = nx/2+1
    mode = "F"
    ctf_2_sum = None

    # create/reset alignment parameters
    for im in range(nima):
        data[im].set_attr( 'ID', list_of_particles[im] )
        util.set_params2D( data[im], [0.0, 0.0, 0.0, 0, 1.0], 'xform.align2d' )
        st = Util.infomask( data[im], mask, False )
        data[im] -= st[0]
        if phase_flip:
            data[im] = filt_ctf(data[im], data[im].get_attr("ctf"), binary = True)

    # precalculate rings & weights
    numr = alignment.Numrinit( first_ring, last_ring, rstep, mode )
    wr   = alignment.ringwe( numr, mode )

    # initialize data for the reference preparation function [the what now?]
    if myid == main_node:
        ref_data = [mask, center, None, None]
        sx_sum = 0.0
        sy_sum = 0.0
        a0 = -1.0e22
        
    recvcount = []
    disp = []
    for i in range(number_of_proc):
        ib, ie = MPI_start_end(total_nima, number_of_proc, i)
        recvcount.append(ie-ib)
        if i == 0:
            disp.append(0)
        else:
            disp.append(disp[i-1]+recvcount[i-1])

    again      = 1
    total_iter = 0
    cs         = [0.0]*2
    delta      = 0.0

    #----------------------------------[ gpu setup ]

    print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "ali2d_base_gpu_isac_CLEAN() :: MPI proc["+str(myid)+"] run pre-alignment GPU setup" )

    # alignment parameters
    aln_cfg = AlignConfig(
        len(data), 1,                # no. of images & averages
        data[0].get_xsize(),         # image size (images are always assumed to be square)
        numr[-3], 256,               # polar sampling params (no. of rings, no. of sample on ring)
        step[0], xrng[0], xrng[0])   # shift params (step size & shift range in x/y dim)
    aln_cfg.freeze()

    # find largest batch size we can fit on the given card
    gpu_batch_limit = 0

    for split in [ 2**i for i in range(int(math.log(len(data),2))+1) ][::-1]:
        aln_cfg.sbj_num = min( gpu_batch_limit + split, len(data) )
        if cu_module.pre_align_size_check( ctypes.c_uint(len(data)), ctypes.byref(aln_cfg), ctypes.c_uint(myid), ctypes.c_float(cuda_device_occ), ctypes.c_bool(False) ) == True:
            gpu_batch_limit += split

    gpu_batch_limit = aln_cfg.sbj_num

    print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "ali2d_base_gpu_isac_CLEAN() :: GPU["+str(myid)+"] pre-alignment batch size: %d/%d" % (gpu_batch_limit, len(data)) )

    # initialize gpu resources (returns location for our alignment parameters in CUDA unified memory)
    gpu_aln_param = cu_module.pre_align_init( ctypes.c_uint(len(data)), ctypes.byref(aln_cfg), ctypes.c_uint(myid) )
    gpu_aln_param = ctypes.cast( gpu_aln_param, aln_param_ptr ) # cast to allow Python-side r/w access

    gpu_batch_count = len(data)/gpu_batch_limit if len(data)%gpu_batch_limit==0 else len(data)//gpu_batch_limit+1

    print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "ali2d_base_gpu_isac_CLEAN() :: GPU["+str(myid)+"] batch count:", gpu_batch_count )

    # if the local stack fits on the gpu we only fetch the img data once before we loop
    if( gpu_batch_count == 1 ):
        cu_module.pre_align_fetch(
            get_c_ptr_array(data),
            ctypes.c_uint(len(data)), 
            ctypes.c_char_p("sbj_batch") )

    #----------------------------------[ alignment ]

    print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "ali2d_base_gpu_isac_CLEAN() :: MPI proc["+str(myid)+"] start alignment iterations" )

    N_step = 0
    # set gpu search parameters
    cu_module.reset_shifts( ctypes.c_float(xrng[N_step]), ctypes.c_float(step[N_step]) )
    
    for Iter in range(max_iter):
        total_iter += 1
        
        #------------------------------------[ construct new average ]
        
        # sum up the original images
        ave1, ave2 = statistics.sum_oe( data, "a", CTF, EMData() )         # pass empty object to prevent calculation of ctf^2
        util.reduce_EMData_to_root(ave1, myid, main_node, comm=mpi_comm)   # sum_oe returns sum of [o]dd and [e]ven items
        util.reduce_EMData_to_root(ave2, myid, main_node, comm=mpi_comm)
        
        if myid == main_node:
            # log iteration info
            log.add("Iteration #%4d"%(total_iter))
            msg = "X range = %5.2f   Y range = %5.2f   Step = %5.2f" % (xrng[N_step], yrng[N_step], step[N_step])
            log.add(msg)
        
            tavg = (ave1+ave2)/total_nima
            
            if outdir:
                tavg.write_image(os.path.join(outdir, "aqc.hdf"), total_iter-1)
                frsc = statistics.fsc_mask(ave1, ave2, mask, 1.0, os.path.join(outdir, "resolution%03d"%(total_iter)))
            else:
                frsc = statistics.fsc_mask(ave1, ave2, mask, 1.0)
        
        else: tavg =  util.model_blank(nx, nx)
        del ave1, ave2
        
        # main node applies (optional) fourier transform and user function (center and filtering)
        if myid == main_node:
            # a0 should increase; stop algorithm when it decreases.    
            #     However, the result will depend on filtration, so it is not quite right.
            #  moved it here, so it is for unfiltered average and thus hopefully makes more sense
            a1 = tavg.cmp("dot", tavg, dict(negative=0, mask=ref_data[0]) )
            msg = "Criterion %d = %15.8e"%(total_iter, a1)
            log.add(msg)
        
            ref_data[2] = tavg
            ref_data[3] = frsc
            # centering (default: average centering method)
            if center == -1:
                ref_data[1] = 0
                tavg, cs = user_func(ref_data)
                cs[0] = float(sx_sum)/total_nima
                cs[1] = float(sy_sum)/total_nima
                tavg = fundamentals.fshift(tavg, -cs[0], -cs[1])
                msg = "Average center x =      %10.3f        Center y       = %10.3f"%(cs[0], cs[1])
                log.add(msg)
            else:
                if delta != 0.0:
                    cnt = ref_data[1]
                    ref_data[1] = 0
                tavg, cs = user_func(ref_data)
                if delta != 0.0:
                    ref_data[1] = cnt
            # write the current filtered average
            if outdir:
                tavg.write_image(os.path.join(outdir, "aqf.hdf"), total_iter-1)
            # update abort criterion
            if a1 < a0:
                if auto_stop:
                    again = 0
            else:
                a0 = a1
        else:
            tavg = util.model_blank(nx, nx)
            cs = [0.0]*2
        
        # check for abort
        if auto_stop:
            again = mpi.mpi_bcast(again, 1, mpi.MPI_INT, main_node, mpi_comm)
        
        #------------------------------------[ alignment ]
        
        # broadcast the newly constructed average to everyone
        util.bcast_EMData_to_all(tavg, myid, main_node, comm=mpi_comm)
        cs = mpi.mpi_bcast(cs, 2, mpi.MPI_FLOAT, main_node, mpi_comm)
        cs = list(map(float, cs))
        
        # backup last iteration's alignment parameters
        old_ali_params = []
        for im in range(nima):  
            alpha, sx, sy, mirror, scale = util.get_params2D(data[im])
            old_ali_params.extend([alpha, sx, sy, mirror])
        ####################################### GPU
        #"""
        # transfer latest average to gpu
        cu_module.pre_align_fetch(
            get_c_ptr_array([tavg]),
            ctypes.c_int(1),
            ctypes.c_char_p("ref_batch") )  # NOTE: happens for each shift change
        
        # FOR gpu_batch_i DO ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
        
        for gpu_batch_idx in range(gpu_batch_count):
        
            # determine next gpu batch and shove to the gpu (NOTE: if we
            # only have a single batch, this already happened earlier)
            if( gpu_batch_count > 1 ):
        
                gpu_batch_start = gpu_batch_idx * gpu_batch_limit
                gpu_batch_end   = gpu_batch_start + gpu_batch_limit
                if gpu_batch_end > len(data): gpu_batch_end = len(data)
        
                cu_module.pre_align_fetch(
                    get_c_ptr_array( data[gpu_batch_start:gpu_batch_end] ),
                    ctypes.c_int( gpu_batch_end-gpu_batch_start ),
                    ctypes.c_char_p("sbj_batch") )
            else:
                gpu_batch_start = 0
                gpu_batch_end   = len(data)

            # run the alignment on gpu
            cu_module.pre_align_run( ctypes.c_int(gpu_batch_start), ctypes.c_int(gpu_batch_end) )
        
            # print progress bar
            gpu_calls_ttl = 1 * max_iter * gpu_batch_count
            #gpu_calls_ttl = len(xrng) * max_iter * gpu_batch_count - 1
            gpu_calls_cnt = N_step*max_iter*gpu_batch_count + Iter*gpu_batch_count + gpu_batch_idx
            gpu_calls_prc = int( float(gpu_calls_cnt+1)/gpu_calls_ttl * 50.0 )
            sys.stdout.write( "\r[PRE-ALIGN][GPU"+str(myid)+"][" + "="*gpu_calls_prc + "-"*(50-gpu_calls_prc) + "]~[%d/%d]~[%.2f%%]" % (gpu_calls_cnt+1, gpu_calls_ttl, (float(gpu_calls_cnt+1)/gpu_calls_ttl)*100.0) )
            sys.stdout.flush()
            if gpu_calls_cnt+1 == gpu_calls_ttl: print("")
        
        # < ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ FOR gpu_batch_i DO
        
        # convert alignment parameters to the format expected by
        # other sphire functions and update EMData headers
        sx_sum, sy_sum = 0.0, 0.0
        
        for k, img in enumerate(data):
            # this is usually done in ormq()
            angle   =  gpu_aln_param[k].angle
            sx_neg  = -gpu_aln_param[k].shift_x
            sy_neg  = -gpu_aln_param[k].shift_y
            c_ang   =  math.cos( math.radians(angle) )
            s_ang   = -math.sin( math.radians(angle) )
            shift_x =  sx_neg*c_ang - sy_neg*s_ang
            shift_y =  sx_neg*s_ang + sy_neg*c_ang
            # this happens in ali2d_single_iter()
            util.set_params2D( img, [angle, shift_x, shift_y, gpu_aln_param[k].mirror, 1.0], "xform.align2d" )
            # this is what is returned by ali2d_single_iter()
            if gpu_aln_param[k].mirror==0:
                sx_sum += shift_x
            else:
                sx_sum -= shift_x
            sy_sum += shift_y
        
        sx_sum = mpi.mpi_reduce( sx_sum, 1, mpi.MPI_FLOAT, mpi.MPI_SUM, main_node, mpi_comm )
        sy_sum = mpi.mpi_reduce( sy_sum, 1, mpi.MPI_FLOAT, mpi.MPI_SUM, main_node, mpi_comm )
        #"""
        ###########################################
        ####################################### GPU
        
        #  for SHC
        if  random_method == "SHC":
            nope = mpi_reduce(nope, 1, mpi.MPI_INT, mpi.MPI_SUM, main_node, mpi_comm)
            nope = mpi_bcast (nope, 1, mpi.MPI_INT, main_node, mpi_comm)
        
        pixel_error       = 0.0
        mirror_consistent = 0
        pixel_error_list  = [-1.0]*nima
        
        for im in range(nima):
            alpha, sx, sy, mirror, scale = util.get_params2D(data[im])
            if old_ali_params[im*4+3] == mirror:
                this_error = pixel_error_2D(old_ali_params[im*4:im*4+3], [alpha, sx, sy], last_ring)
                pixel_error += this_error
                pixel_error_list[im] = this_error
                mirror_consistent += 1
        del old_ali_params
        
        mirror_consistent = mpi.mpi_reduce(mirror_consistent, 1, mpi.MPI_INT, mpi.MPI_SUM, main_node, mpi_comm)
        pixel_error       = mpi.mpi_reduce(pixel_error, 1, mpi.MPI_FLOAT, mpi.MPI_SUM, main_node, mpi_comm)
        pixel_error_list  = mpi.mpi_gatherv(pixel_error_list, nima, mpi.MPI_FLOAT, recvcount, disp, mpi.MPI_FLOAT, main_node, mpi_comm)

    if myid == main_node and outdir:
        util.drop_image(tavg, os.path.join(outdir, "aqfinal.hdf"))

    # free gpu resources
    cu_module.gpu_clear()
    mpi.mpi_barrier(mpi_comm)

    params = []
    for im in range(nima):  
        alpha, sx, sy, mirror, scale = util.get_params2D(data[im])
        params.append([alpha, sx, sy, mirror])

    mpi.mpi_barrier(mpi_comm)
    tmp = params[:]
    tmp = spx.wrap_mpi_gatherv(tmp, main_node, mpi_comm)
    if( myid == main_node ):
        spx.write_text_row( tmp, os.path.join(outdir,"initial2Dparams.txt") )
    del tmp

    if myid == main_node:
        log.add("Finished ali2d_base")

    if myid==0: print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "ali2d_base_gpu_isac_CLEAN() :: Alignment complete" )

    return params

def ali2d_base(stack, outdir, maskfile=None, ir=1, ou=-1, rs=1, xr="4 2 1 1", yr="-1", ts="2 1 0.5 0.25", \
            nomirror = False, dst=0.0, center=-1, maxit=0, CTF=False, snr=1.0, \
            Fourvar=False, user_func_name="ref_ali2d", random_method = "", log = None, \
            number_of_proc = 1, myid = 0, main_node = 0, mpi_comm = None, write_headers = False):

    from sp_utilities    import model_circle, model_blank, drop_image, get_image, get_input_from_string
    from sp_utilities    import reduce_EMData_to_root, bcast_EMData_to_all, send_attr_dict, file_type
    from sp_utilities    import bcast_number_to_all, bcast_list_to_all
    from sp_statistics   import fsc_mask, sum_oe, hist_list, varf2d_MPI
    from sp_alignment    import Numrinit, ringwe, ali2d_single_iter
    from sp_pixel_error  import pixel_error_2D
    from numpy        import reshape, shape
    from sp_fundamentals import fshift, fft, rot_avg_table
    from sp_utilities    import get_params2D, set_params2D
    from sp_utilities    import wrap_mpi_gatherv
    import os
    import sys
    from sp_applications   import MPI_start_end
    from mpi       import mpi_init, mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD
    from mpi       import mpi_reduce, mpi_bcast, mpi_barrier, mpi_gatherv
    from mpi       import MPI_SUM, MPI_FLOAT, MPI_INT
    import time

    if log == None:
        from sp_logger import Logger
        log = Logger()

    if mpi_comm == None:
        mpi_comm = MPI_COMM_WORLD

    # ftp = file_type(stack)

    if myid == main_node:
        import sp_global_def
        sp_global_def.LOGFILE =  os.path.join(outdir, sp_global_def.LOGFILE)
        log.add("Start  ali2d_MPI")

    xrng        = get_input_from_string(xr)
    if  yr == "-1":  yrng = xrng
    else          :  yrng = get_input_from_string(yr)
    step        = get_input_from_string(ts)
    
    first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit);
    
    if max_iter == 0:
        max_iter = 10
        auto_stop = True
    else:
        auto_stop = False
    
    import types
    if( type(stack) is bytes ):
        if myid == main_node:
            total_nima = EMUtil.get_image_count(stack)
        else:
            total_nima = 0
        total_nima = bcast_number_to_all(total_nima)
        list_of_particles = list(range(total_nima))

        image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
        list_of_particles = list_of_particles[image_start:image_end]
        nima = len(list_of_particles)
        data = EMData.read_images(stack, list_of_particles)

    else:
        data = stack
        total_nima = len(data)
        total_nima = mpi_reduce(total_nima, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD)
        total_nima = mpi_bcast(total_nima, 1, MPI_INT, main_node, MPI_COMM_WORLD)[0]
        list_of_particles = list(range(total_nima))
        image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
        list_of_particles = list_of_particles[image_start:image_end]
        nima = len(list_of_particles)
        assert( nima == len(data))

    # read nx and ctf_app (if CTF) and broadcast to all nodes
    if myid == main_node:
        nx = data[0].get_xsize()
    else:
        nx = 0

    nx = bcast_number_to_all(nx, source_node = main_node)

    phase_flip = False
    CTF = False

    # default value for the last ring
    if last_ring == -1: last_ring = nx/2-2

    if last_ring + max([max(xrng), max(yrng)]) > (nx-1) // 2:
        ERROR('Shift or radius is too large - particle crosses image boundary', "ali2d_MPI", 1)
    
    if myid == main_node:
        # log.add("Input stack                 : %s"%(stack))
        log.add("Number of images            : %d"%(total_nima))
        log.add("Output directory            : %s"%(outdir))
        log.add("Inner radius                : %i"%(first_ring))
        log.add("Outer radius                : %i"%(last_ring))
        log.add("Ring step                   : %i"%(rstep))
        log.add("X search range              : %s"%(xrng))
        log.add("Y search range              : %s"%(yrng))
        log.add("Translational step          : %s"%(step))
        log.add("Disable checking mirror     : %s"%(nomirror))
        log.add("Discrete angle used         : %d"%(dst))
        log.add("Center type                 : %i"%(center))
        log.add("Maximum iteration           : %i"%(max_iter))
        #log.add("Use Fourier variance        : %s\n"%(Fourvar))
        log.add("CTF correction              : %s"%(CTF))
        log.add("Phase flip                  : %s"%(phase_flip))
        #log.add("Signal-to-Noise Ratio       : %f\n"%(snr))
        if auto_stop:
            log.add("Stop iteration with         : criterion")
        else:
            log.add("Stop iteration with         : maxit")

        import sp_user_functions
        user_func = sp_user_functions.factory[user_func_name]

        log.add("User function               : %s"%(user_func_name))
        log.add("Number of processors used   : %d"%(number_of_proc))

    if maskfile:
        import  types
        if type(maskfile) is bytes:  
            if myid == main_node:        log.add("Maskfile                    : %s"%(maskfile))
            mask = get_image(maskfile)
        else:
            if myid == main_node:         log.add("Maskfile                    : user provided in-core mask")
            mask = maskfile
    else:
        if myid == main_node:     log.add("Maskfile                    : default, a circle with radius %i"%(last_ring))
        mask = model_circle(last_ring, nx, nx)

    cnx  = nx/2+1
    cny  = cnx
    if  random_method == "SCF":        mode = "H"
    else:                             mode = "F"

    ctf_2_sum = None

    for im in range(nima):
        data[im].set_attr('ID', list_of_particles[im])
        set_params2D(data[im], [0.0, 0.0, 0.0, 0, 1.0], 'xform.align2d')
        st = Util.infomask(data[im], mask, False)
        data[im] -= st[0]
        if( random_method == "SHC" ):  data[im].set_attr('previousmax',1.0e-23)

    ctf_2_sum = None

    # startup
    numr = Numrinit(first_ring, last_ring, rstep, mode)     #precalculate rings
    wr = ringwe(numr, mode)

    if myid == main_node:
        # initialize data for the reference preparation function
        ref_data = [mask, center, None, None]
        sx_sum = 0.0
        sy_sum = 0.0
        a0 = -1.0e22
        
    recvcount = []
    disp = []
    for i in range(number_of_proc):
        ib, ie = MPI_start_end(total_nima, number_of_proc, i)
        recvcount.append(ie-ib)
        if i == 0:
            disp.append(0)
        else:
            disp.append(disp[i-1]+recvcount[i-1])


    N_step = 0 #only test first
    again = 1
    total_iter = 0
    cs = [0.0]*2
    delta = 0.0
    if myid == main_node:  print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "ali2d_base()  start alignment iterations" )
    for Iter in range(max_iter):
        
        total_iter += 1
        ave1, ave2 = sum_oe(data, "a", CTF, EMData())  # pass empty object to prevent calculation of ctf^2
        reduce_EMData_to_root(ave1, myid, main_node)
        reduce_EMData_to_root(ave2, myid, main_node)
        sys.stdout.flush()
        if myid == main_node:
            log.add("Iteration #%4d"%(total_iter))
            msg = "X range = %5.2f   Y range = %5.2f   Step = %5.2f"%(xrng[N_step], yrng[N_step], step[N_step])
            log.add(msg)
            tavg = (ave1+ave2)/total_nima
            if outdir:
                tavg.write_image(os.path.join(outdir, "aqc.hdf"), total_iter-1)
                frsc = fsc_mask(ave1, ave2, mask, 1.0, os.path.join(outdir, "resolution%03d"%(total_iter)))
            else:
                frsc = fsc_mask(ave1, ave2, mask, 1.0)
        else:
            tavg =  model_blank(nx, nx)
        del ave1, ave2
            
        if Fourvar:  
            bcast_EMData_to_all(tavg, myid, main_node)
            vav, rvar = varf2d_MPI(myid, data, tavg, mask, "a", CTF)
        
        if myid == main_node:
            if Fourvar:
                tavg    = fft(Util.divn_img(fft(tavg), vav))
                vav_r    = Util.pack_complex_to_real(vav)
                if outdir:
                    vav_r.write_image(os.path.join(outdir, "varf.hdf"), total_iter-1)
        
        
            # a0 should increase; stop algorithm when it decreases.    
            #     However, the result will depend on filtration, so it is not quite right.
            #  moved it here, so it is for unfiltered average and thus hopefully makes more sense
            a1 = tavg.cmp("dot", tavg, dict(negative = 0, mask = ref_data[0]))
            msg = "Criterion %d = %15.8e"%(total_iter, a1)
            log.add(msg)
        
        
            ref_data[2] = tavg
            ref_data[3] = frsc
        
            #  call user-supplied function to prepare reference image, i.e., center and filter it
            if center == -1:
                # When center = -1, which is by default, we use the average center method
                ref_data[1] = 0
                tavg, cs = user_func(ref_data)
                cs[0] = float(sx_sum)/total_nima
                cs[1] = float(sy_sum)/total_nima
                tavg = fshift(tavg, -cs[0], -cs[1])
                msg = "Average center x =      %10.3f        Center y       = %10.3f"%(cs[0], cs[1])
                log.add(msg)
            else:
                if delta != 0.0:
                    cnt = ref_data[1]
                    ref_data[1] = 0
                tavg, cs = user_func(ref_data)
                if delta != 0.0:
                    ref_data[1] = cnt
            # write the current filtered average
            if outdir:
                tavg.write_image(os.path.join(outdir, "aqf.hdf"), total_iter-1)
        
            if a1 < a0:
                if auto_stop:     again = 0
            else:    a0 = a1
        else:
            tavg = model_blank(nx, nx)
            cs = [0.0]*2
        
        if auto_stop:
            again = mpi_bcast(again, 1, MPI_INT, main_node, mpi_comm)
        
        if Fourvar:  del vav
        bcast_EMData_to_all(tavg, myid, main_node)
        cs = mpi_bcast(cs, 2, MPI_FLOAT, main_node, mpi_comm)
        cs = list(map(float, cs))
        if total_iter != max_iter*len(xrng):
            old_ali_params = []
            for im in range(nima):  
                alpha, sx, sy, mirror, scale = get_params2D(data[im])
                old_ali_params.extend([alpha, sx, sy, mirror])

            if Iter%4 != 0 or total_iter > max_iter*len(xrng)-10: delta = 0.0
            else: delta = dst        

            sx_sum, sy_sum, nope = ali2d_single_iter(data, numr, wr, cs, tavg, cnx, cny, \
                                            xrng[N_step], yrng[N_step], step[N_step], \
                                            nomirror=nomirror, mode=mode, CTF=CTF, delta=delta, \
                                            random_method = random_method)
        
            sx_sum = mpi_reduce(sx_sum, 1, MPI_FLOAT, MPI_SUM, main_node, mpi_comm)
            sy_sum = mpi_reduce(sy_sum, 1, MPI_FLOAT, MPI_SUM, main_node, mpi_comm)
            #  for SHC
            if  random_method == "SHC":
                nope   = mpi_reduce(nope, 1, MPI_INT, MPI_SUM, main_node, mpi_comm)
                nope   = mpi_bcast(nope, 1, MPI_INT, main_node, mpi_comm)
        
            pixel_error       = 0.0
            mirror_consistent = 0
            pixel_error_list  = [-1.0]*nima
            for im in range(nima):
                alpha, sx, sy, mirror, scale = get_params2D(data[im])
                
                if old_ali_params[im*4+3] == mirror:
                    this_error = pixel_error_2D(old_ali_params[im*4:im*4+3], [alpha, sx, sy], last_ring)
                    pixel_error += this_error
                    pixel_error_list[im] = this_error
                    mirror_consistent += 1
            del old_ali_params
            mirror_consistent = mpi_reduce(mirror_consistent, 1, MPI_INT, MPI_SUM, main_node, mpi_comm)
            pixel_error       = mpi_reduce(pixel_error, 1, MPI_FLOAT, MPI_SUM, main_node, mpi_comm)
            pixel_error_list  = mpi_gatherv(pixel_error_list, nima, MPI_FLOAT, recvcount, disp, MPI_FLOAT, main_node, mpi_comm)
            if myid == main_node:
                log.add("Mirror consistency rate = %8.4f%%"%(float(mirror_consistent)/total_nima*100))
                if mirror_consistent!=0:
                    log.add("Among the mirror-consistent images, average of pixel errors is %0.4f, and their distribution is:"%(float(pixel_error)/float(mirror_consistent)))
                    pixel_error_list = list(map(float, pixel_error_list))
                    for i in range(total_nima-1, -1, -1):
                        if pixel_error_list[i] < 0:  del pixel_error_list[i]
                    region, hist = hist_list(pixel_error_list, 20)
                    for p in range(20):
                        log.add("      %14.2f: %6d"%(region[p], hist[p]))
                log.add("\n\n")

    if myid == main_node and outdir:  drop_image(tavg, os.path.join(outdir, "aqfinal.hdf"))
    if myid == main_node:  print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "ali2d_base()  end alignment iterations" )
    # write out headers and STOP, under MPI writing has to be done sequentially
    mpi_barrier(mpi_comm)
    params = []
    for im in range(nima):  
        alpha, sx, sy, mirror, scale = get_params2D(data[im])
        params.append([alpha, sx, sy, mirror])
        
    mpi_barrier(mpi_comm)
    tmp = params[:]
    tmp = spx.wrap_mpi_gatherv(tmp, main_node, mpi_comm)
    if( myid == main_node ):
        spx.write_text_row( tmp, os.path.join(outdir,"initial2Dparams.txt") )
    del tmp
    
    if myid == main_node: log.add("Finished ali2d_base")

    return params #, data

def mpi_assert( condition, msg ):
    if not condition:
        mpi_rank = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
        print( "MPI PROC["+str(mpi_rank)+"] ASSERTION ERROR:", msg, file=sys.stderr)
        sys.stderr.flush()
        mpi.mpi_finalize()
        sys.exit()

def main():
    progname = os.path.basename(sys.argv[0])
    usage = progname + " stack outdir <maskfile> --ir=inner_radius --ou=outer_radius --rs=ring_step --xr=x_range --yr=y_range --ts=translation_step --dst=delta --center=center --maxit=max_iteration --CTF --snr=SNR --Fourvar=Fourier_variance --Ng=group_number --Function=user_function_name --CUDA --GPUID --MPI"
    parser = OptionParser(usage,version=SPARXVERSION)
    parser.add_option("--ir",       type="float",  default=1,             help="inner radius for rotational correlation > 0 (set to 1)")
    parser.add_option("--ou",       type="float",  default=-1,            help="outer radius for rotational correlation < nx/2-1 (set to the radius of the particle)")
    parser.add_option("--rs",       type="float",  default=1,             help="step between rings in rotational correlation > 0 (set to 1)" ) 
    parser.add_option("--xr",       type="string", default="4 2 1 1",     help="range for translation search in x direction, search is +/xr ")
    parser.add_option("--yr",       type="string", default="-1",          help="range for translation search in y direction, search is +/yr ")
    parser.add_option("--ts",       type="string", default="2 1 0.5 0.25",help="step of translation search in both directions")
    parser.add_option("--nomirror", action="store_true", default=False,   help="Disable checking mirror orientations of images (default False)")
    parser.add_option("--dst",      type="float",  default=0.0,           help="delta")
    parser.add_option("--center",   type="float",  default=-1,            help="-1.average center method; 0.not centered; 1.phase approximation; 2.cc with Gaussian function; 3.cc with donut-shaped image 4.cc with user-defined reference 5.cc with self-rotated average")
    parser.add_option("--maxit",    type="float",  default=0,             help="maximum number of iterations (0 means the maximum iterations is 10, but it will automatically stop should the criterion falls")
    parser.add_option("--CTF",      action="store_true", default=False,   help="use CTF correction during alignment")
    parser.add_option("--snr",      type="float",  default=1.0,           help="signal-to-noise ratio of the data (set to 1.0)")
    parser.add_option("--Fourvar",  action="store_true", default=False,   help="compute Fourier variance")
    parser.add_option("--function", type="string",       default="ref_ali2d",  help="name of the reference preparation function (default ref_ali2d)")
    parser.add_option("--gpu_devices",     type="string",       default="",    help="Specify the GPUs to be used (e.g. --gpu_devices=0, or --gpu_devices=0,1 for one or two GPUs, respectively). Using \"$ nividia-smi\" in the terminal will print out what GPUs are available. [Default: None]" )
    parser.add_option( "--gpu_info",         action="store_true", default=False, help="Print detailed information about the selected GPUs. Use --gpu_devices to specify what GPUs you want to know about. NOTE: program will stop after printing this information, so don't use this parameter if you intend to actually process any data. [Default: False]" )
    parser.add_option("--MPI",      action="store_true", default=False,   help="use MPI version ")
    parser.add_option("--mode",     type="string",       default="F",     help="Full or Half rings, default F")
    parser.add_option("--randomize",action="store_true", default=False,   help="randomize initial rotations (suboption of friedel, default False)")
    parser.add_option("--orient",   action="store_true", default=False,   help="orient images such that the average is symmetric about x-axis, for layer lines (suboption of friedel, default False)")
    parser.add_option("--random_method",   type="string", default="",   help="use SHC or SCF (default standard method)")

    (options, args) = parser.parse_args()

    if len(args) < 2 or len(args) > 3:
        print("usage: " + usage)
        print("Please run '" + progname + " -h' for detailed options")
    else:
        if args[1] == 'None': outdir = None
        else:                 outdir = args[1]

        if len(args) == 2: mask = None
        else:              mask = args[2]

        if global_def.CACHE_DISABLE:
            from utilities import disable_bdb_cache
            disable_bdb_cache()
        global_def.BATCH = True

        if  options.MPI:
            from mpi import mpi_init, mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD
            import mpi
            sys.argv = mpi_init(len(sys.argv),sys.argv) # init and finalize are needed for each process

            number_of_proc = mpi_comm_size(MPI_COMM_WORLD) # read from mpirun
            myid = mpi_comm_rank(MPI_COMM_WORLD)
            main_node = 0

            if(myid == main_node):
                import subprocess
                from logger import Logger, BaseLogger_Files
                #  Create output directory
                log = Logger(BaseLogger_Files())
                log.prefix = os.path.join(outdir)
                cmd = "mkdir "+log.prefix
                outcome = subprocess.call(cmd, shell=True)
                log.prefix += "/"
            else:
                outcome = 0
                log = None
            from utilities       import bcast_number_to_all
            outcome  = bcast_number_to_all(outcome, source_node = main_node)
            if(outcome == 1):
                ERROR('Output directory exists, please change the name and restart the program', "ali2d_MPI", 1, myid)

            params2d = ali2d_base(args[0], outdir, mask, options.ir, options.ou, options.rs, options.xr, options.yr, \
                options.ts, options.nomirror, options.dst, \
                options.center, options.maxit, options.CTF, options.snr, options.Fourvar, \
                options.function, random_method = options.random_method, log = log, \
                number_of_proc = number_of_proc, myid = myid, main_node = main_node, mpi_comm = MPI_COMM_WORLD,\
                write_headers = True)

        else:
            import time
            from mpi import mpi_init, mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD
            import mpi
            sys.argv = mpi_init(len(sys.argv),sys.argv)
            global Blockdata
            Blockdata = {}
            Blockdata["nproc"]        = mpi_comm_size(mpi.MPI_COMM_WORLD)
            myid = mpi_comm_rank(MPI_COMM_WORLD)
            main_node = 0
            ### GPU Related ###
            Blockdata["shared_comm"]  = mpi.mpi_comm_split_type(mpi.MPI_COMM_WORLD, mpi.MPI_COMM_TYPE_SHARED, 0, mpi.MPI_INFO_NULL)
            Blockdata["myid_on_node"] = mpi.mpi_comm_rank(Blockdata["shared_comm"])                                                                                                              
                                                                                                              
            tmp_img = EMData() # this is just a placeholder EMData object that we'll re-use in a couple of loops                                                                                                  
            # get total number of images (nima) and broadcast
            if(myid == main_node): 
                Blockdata["total_nima"] = global_def.EMUtil.get_image_count(args[0])
            else: 
                Blockdata["total_nima"] = 0

            Blockdata["total_nima"] = spx.bcast_number_to_all(Blockdata["total_nima"], source_node = main_node)
            
            #####################  [ System and GPU Memory check ]      #####################      
            # map our GPU selection to list of available GPUs
            global GPU_DEVICES
            if options.gpu_devices != "":
                os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu_devices
                options.gpu_devices = ",".join( map(str, range(len(options.gpu_devices.split(",")))) )
                GPU_DEVICES = map( int, options.gpu_devices.split(",") )
                if myid==0: print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "main() :: Using CUDA devices", GPU_DEVICES )
            else:
                GPU_DEVICES = [0]
                os.environ["CUDA_VISIBLE_DEVICES"] = '0'
                if myid==0: 
                    print( "\nWarning: No GPU was specified. Using GPU [0] by default.")
                    print( "           -> Program will crash if the selected GPU does not suffice." )
                    print( "           -> Use \"$ nividia-smi\" in the terminal to see a list of available GPUs.\'\n" )
            # sanity check: make sure each GPU can be assigned to an MPI process
            mpi_assert( len(GPU_DEVICES) <= Blockdata["nproc"], "ERROR! Trying to use more GPUs (x"+str(len(GPU_DEVICES))+") than MPI are available (x"+str(Blockdata["nproc"])+")!" )
            
            # GPU info
            if options.gpu_info:
                if myid==0:
                    import applications
                    for cuda_device_id in GPU_DEVICES:
                        print( "\n____[ CUDA device("+str(cuda_device_id)+") ]____" )
                        applications.print_gpu_info(cuda_device_id)
                        sys.stdout.flush()
                mpi.mpi_finalize()
                sys.exit()
            
            global MPI_GPU_COMM
            MPI_GPU_COMM = mpi.mpi_comm_split( mpi.MPI_COMM_WORLD, (Blockdata["myid_on_node"] in GPU_DEVICES), myid )
            #------------------------------------------------------[ Memory check ]

            # percentage of system memory we allow ourselves to occupy; we leave some
            # in over to leave gpu isac and others some breathing room
            sys_mem_use = 0.75
            
            # we use a linux system call to get the RAM info we need
            if "linux" in sys.platform:
                if myid==0: print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "main() :: Running memory check.." )
            
                # get amount of available system RAM
                sys_mem_avl = float( os.popen("free").readlines()[1].split()[6] ) / 1024 # mem size in MB (free returns kb)
                if myid==0:
                    print( " "*32, "..avl sys mem (RAM): %.2f MB / %.2f GB" % (sys_mem_avl, sys_mem_avl/1024) )
                    print( " "*32, "..use sys mem (RAM): %.2f MB / %.2f GB" % (sys_mem_avl*sys_mem_use, (sys_mem_avl*sys_mem_use)/1024) )
            
                # estimate memory requirement of raw data
                tmp_img.read_image( args[0], 0 )
                data_size_sub = (Blockdata["total_nima"]*tmp_img.get_xsize()*tmp_img.get_ysize()*4.) / 1024**2 # mem size in MB
                if myid==0: print( " "*32, "..est mem req (raw data, %d %dx%d images): %.2f MB / %.2f GB" %
                    (Blockdata["total_nima"], tmp_img.get_xsize(), tmp_img.get_ysize(),
                     data_size_sub, data_size_sub/1024) )
            
                # batch size of input reads per MPI proc
                batch_mem_avl = (sys_mem_avl*sys_mem_use - data_size_sub) / Blockdata["nproc"]
                batch_img_num = int( batch_mem_avl / (data_size_sub/Blockdata["total_nima"]) )
                if myid==0: print( " "*32, "..%d MPI procs set to read data in batches of (max) %d images each (batch mem: %.2f MB / %.2f GB)" %
                      (Blockdata["nproc"], batch_img_num, batch_mem_avl, batch_mem_avl/1024) )
                mpi_assert( batch_img_num > 0, "Memory cannot even! batch_img_num is %d"%batch_img_num )
            
                # make sure we can keep the downsampled data in RAM
                if data_size_sub > sys_mem_avl*sys_mem_use and data_size_sub < sys_mem_avl:
                    if myid==0: print(" "*32, ">>WARNING. Requested job requires almost all available system RAM." )
                elif data_size_sub > sys_mem_avl:
                    if myid==0: print(" "*32, ">>ERROR. Requested job will not fit into available system RAM!" )
                else:
                    if myid==0: print(" "*32, ">>All good to go!" )
            else:
                if myid==0: print( "WARNING! Running on unsupported platform. No memory check was performed." )
                
            #####################  [ System and GPU Memory check ]      #####################      
            
            
            
            if(myid == main_node):
                import subprocess
                from logger import Logger, BaseLogger_Files
                #  Create output directory
                log = Logger(BaseLogger_Files())
                log.prefix = os.path.join(outdir)
                cmd = "mkdir "+log.prefix
                outcome = subprocess.call(cmd, shell=True)
                log.prefix += "/"
            else:
                outcome = 0
                log = None
                
            from utilities       import bcast_number_to_all
            outcome  = bcast_number_to_all(outcome, source_node = main_node)
            if(outcome == 1):
                ERROR('Output directory exists, please change the name and restart the program', "ali2d_MPI", 1, myid)

            # make extra double sure the file system has caught up
            mpi.mpi_barrier( mpi.MPI_COMM_WORLD )
            while not os.path.exists(os.path.join(outdir)+"/"):
                time.sleep(1)
            mpi.mpi_barrier( mpi.MPI_COMM_WORLD )
            
            #--------------------------------------------------[ first data read ]
            
            if myid==0: print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "main() :: Reading data" )
            
            # local setup
            image_start, image_end = spx.MPI_start_end( Blockdata["total_nima"], Blockdata["nproc"], myid )
            original_images = [None] * (image_end-image_start)

            
            # all MPI procs: read and immediately process/resample data batches
            idx=0
            batch_start = image_start
            while batch_start < image_end:
                batch_end = min( batch_start+batch_img_num, image_end )
                # read batch
                for i, img_idx in enumerate( range(batch_start, batch_end) ):
                    tmp_img.read_image( args[0], img_idx )
                    original_images[idx+i] = tmp_img
                # go to next batch
                batch_start += batch_img_num
                idx += batch_img_num

            mpi.mpi_barrier( mpi.MPI_COMM_WORLD ) # just to print msg below after the progress bars above
            if myid==0: print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "main() :: Distributing workload to available GPUs" )

            #--------------------------------------------------[ collect data from all procs in GPU procs ]

            # step 01: everyone sends their data to the GPU looking for it
            for gpu in GPU_DEVICES:
                gpu_img_start, gpu_img_end = spx.MPI_start_end( Blockdata["total_nima"], len(GPU_DEVICES), gpu )

                if gpu==myid:
                    continue

                for i, my_img in enumerate( range(image_start, image_end) ):
                    if gpu_img_start <= my_img < gpu_img_end:
                        spx.send_EMData( original_images[i], gpu, my_img, comm=mpi.MPI_COMM_WORLD )
                        original_images[i] = None

            original_images = [ i for i in original_images if i is not None ] # might be a GPU process sends some and keeps the rest

            # step 02a: each GPU proc receives the desired data as offered by the other processes
            if myid in GPU_DEVICES:
                image_start, image_end = spx.MPI_start_end( Blockdata["total_nima"], len(GPU_DEVICES), myid )

                for proc in range( Blockdata["nproc"] ):
                    proc_img_start, proc_img_end = spx.MPI_start_end( Blockdata["total_nima"], Blockdata["nproc"], proc )

                    if proc==myid:
                        continue

                    for proc_img in range( proc_img_start, proc_img_end ):
                        if image_start <= proc_img < image_end:
                            original_images.append( spx.recv_EMData(proc, proc_img, comm=mpi.MPI_COMM_WORLD) )

            # step 2b: each non-GPU proc makes sure they sent off all their data
            else:
                assert len(original_images) == 0, "ERROR: proc[%d] still holds %d images." % (myid, len(original_images))
                image_start, image_end = None, None

            mpi.mpi_barrier(mpi.MPI_COMM_WORLD) # the above communication is blocking, but just to be sure

            #--------------------------------------------------[ run the GPU pre-alignment ]
            
            if Blockdata["myid_on_node"] in GPU_DEVICES:
            
                # 2D gpu alignment call
                if myid==0: print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "main() :: Executing pre-alignment" )

                params2d = ali2d_base_gpu_isac_CLEAN(
                    original_images,                  # downsampled images handled by this process only
                    outdir,                         # output directory
                    mask,                             # mask
                    options.ir,                                # inner radius / first ring
                    options.ou,                    # outer radius / last ring
                    options.rs,                                # ring step
                    options.xr,                              # list of search rangees in x-dim
                    options.yr,                              # list of search rangees in y-dim
                    options.ts,                              # search step size
                    options.nomirror,                            # no mirror flag
                    options.dst,                             # alignment angle reset value
                    options.center,                    # centering method (should be 0)
                    options.maxit,                               # iteration limit
                    options.CTF,                      # CTF flag
                    options.snr,                              # snr (CTF parameter)
                    options.Fourvar,                            # some fourier flag?
                    options.function,                      # user_func_name(?)
                    options.random_method,                               # randomization method
                    log,                            # log (?)           _
                    mpi.mpi_comm_size(MPI_GPU_COMM),  # mpi comm size      |
                    mpi.mpi_comm_rank(MPI_GPU_COMM),  # mpi rank           |_______[ gpu communicator ]
                    0,                                # mpi main proc      |
                    MPI_GPU_COMM,                     # mpi communicator  _|
                    write_headers=False,              # we write the align params to a file (below) but not to particle headers
                    mpi_gpu_proc=(Blockdata["myid_on_node"] in GPU_DEVICES),
                    cuda_device_occ=0.9 )
                
                mpi.mpi_barrier( MPI_GPU_COMM)
                
                if myid==0: print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "main() :: Pre-alignment call complete" )

        global_def.BATCH = False
        mpi.mpi_finalize()

if __name__ == "__main__":
    main()
