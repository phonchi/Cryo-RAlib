#!/usr/local/EMAN2/bin/python
from __future__ import print_function

import os
import global_def
from global_def import *
from optparse import OptionParser
import sys
import sparx as spx

###############################################################
########################################################### GPU

import math
import numpy as np
import ctypes
from EMAN2 import EMNumPy
from cupy.cuda.nvtx import RangePush, RangePushC, RangePop

CUDA_PATH = os.path.join(os.path.dirname(__file__),  "cuda", "")
cu_module = ctypes.CDLL( CUDA_PATH + "gpu_aln_pack.so" )
float_ptr = ctypes.POINTER(ctypes.c_float)

cu_module.ref_free_alignment_2D_init.restype = ctypes.c_ulonglong
cu_module.pre_align_init.restype = ctypes.c_ulonglong
cu_module.gpu_image_ptr.restype = ctypes.POINTER(ctypes.c_float)

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

def send_EMData_w_ID(img, dst, tag, comm=-1):
    from mpi import mpi_send, MPI_INT, MPI_FLOAT, MPI_COMM_WORLD
    from sp_utilities import get_image_data

    if comm==-1:
        comm = MPI_COMM_WORLD

    img_head = []
    img_head.append(img.get_xsize())
    img_head.append(img.get_ysize())
    img_head.append(img.get_zsize())
    img_head.append(img.is_complex())
    img_head.append(img.is_ri())
    img_head.append(img.get_attr("changecount"))
    img_head.append(img.is_complex_x())
    img_head.append(img.get_attr("is_complex_ri"))
    img_head.append(int(img.get_attr("apix_x")*10000))
    img_head.append(int(img.get_attr("apix_y")*10000))
    img_head.append(int(img.get_attr("apix_z")*10000))
    img_head.append(img.get_attr("ID"))

    head_tag = 2*tag
    mpi_send(img_head, 12, MPI_INT, dst, head_tag, comm)

    img_data = get_image_data(img)
    data_tag = 2*tag + 1
    ntot = img_head[0]*img_head[1]*img_head[2]
    mpi_send(img_data, ntot, MPI_FLOAT, dst, data_tag, comm)

def recv_EMData_w_ID(src, tag, comm=-1):
    from mpi import mpi_recv, MPI_INT, MPI_FLOAT, MPI_COMM_WORLD
    from numpy import reshape
    from EMAN2 import EMNumPy

    if comm==-1:
        comm = MPI_COMM_WORLD

    head_tag = 2*tag
    img_head = mpi_recv(12, MPI_INT, src, head_tag, comm)

    nx = int(img_head[0])
    ny = int(img_head[1])
    nz = int(img_head[2])
    is_complex = int(img_head[3])
    is_ri = int(img_head[4])

    data_tag = 2*tag+1
    ntot = nx*ny*nz

    img_data = mpi_recv(ntot, MPI_FLOAT, src, data_tag, comm)
    if nz != 1:
        img_data = reshape(img_data, (nz, ny, nx))
    elif ny != 1:
        img_data = reshape(img_data, (ny, nx))
    else:
        pass

    img = EMNumPy.numpy2em(img_data)
    img.set_complex(is_complex)
    img.set_ri(is_ri)
    img.set_attr_dict({"changecount":int(img_head[5]),  
                        "is_complex_x":int(img_head[6]),  
                        "is_complex_ri":int(img_head[7]),  
                        "apix_x":int(img_head[8])/10000.0,  
                        "apix_y":int(img_head[9])/10000.0,  
                        "apix_z":int(img_head[10])/10000.0,
                        "ID":int(img_head[11])})
    return img

########################################################### GPU
###############################################################
def mref_ali2d_gpu(
    filename,stack, refim, outdir, maskfile=None, ir=1, ou=-1, rs=1, xrng = 0, yrng = 0, step = 1,
    center=-1, maxit=0, CTF=False, snr=1.0,
    user_func_name="ref_ali2d", rand_seed= 1000, number_of_proc = 1, myid = 0, main_node = 0, mpi_comm = None, 
    mpi_gpu_proc=False, gpu_class_limit=0, cuda_device_occ=0.9):

    from sp_utilities      import   model_circle, combine_params2, inverse_transform2, drop_image, get_image, get_im
    from sp_utilities      import   reduce_EMData_to_root, bcast_EMData_to_all, bcast_number_to_all
    from sp_utilities      import   send_attr_dict
    from sp_utilities        import   center_2D
    from sp_statistics     import   fsc_mask
    from sp_alignment      import   Numrinit, ringwe, search_range
    from sp_fundamentals   import   rot_shift2D, fshift
    from sp_utilities      import   get_params2D, set_params2D
    from random         import   seed, randint
    from sp_morphology     import   ctf_2
    from sp_filter         import   filt_btwl, filt_params
    from numpy          import   reshape, shape
    from sp_utilities      import   print_msg, print_begin_msg, print_end_msg
    import os
    import alignment
    import fundamentals
    import sys
    import mpi
    import time
    import utilities as util
    from sp_applications   import MPI_start_end
    from mpi       import mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD
    from mpi       import mpi_reduce, mpi_bcast, mpi_barrier, mpi_recv, mpi_send
    from mpi       import MPI_SUM, MPI_FLOAT, MPI_INT

    import sp_user_functions
    from sp_applications   import MPI_start_end
    user_func = sp_user_functions.factory[user_func_name]

    # sanity check: gpu procs sound off
    if not mpi_gpu_proc: return []


    #----------------------------------[ setup ]
    print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "mref_ali2d_gpu() :: MPI proc["+str(myid)+"] run pre-alignment CPU setup" )

    # mpi communicator sanity check
    assert mpi_comm != None
    
    xrng = [xrng]
    yrng = [yrng]
    step = [step]
    # polar conversion (ring) parameters
    first_ring = int(ir)
    last_ring  = int(ou)
    rstep      = int(rs)
    max_iter   = int(maxit)

    if max_iter ==0:
        max_iter = 10
        auto_stop = True
    else:
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
    if last_ring + max([max(xrng),max(yrng)]) > (nx-1) // 2:
        ERROR( "Shift or radius is too large - particle crosses image boundary", "ali2d_MPI", 1 )

    if maskfile:
        import  types
        if type(maskfile) is bytes:  mask = get_image(maskfile)
        else: mask = maskfile
    else : mask = model_circle(last_ring, nx, nx)

    # references, do them on all processors
    refi = []
    numref = EMUtil.get_image_count(refim)
    # image center
    cny = cnx = nx/2+1
    mode = "F"
    RangePush("Preprocess data")
    # prepare reference images on all nodes
    ima = data[0].copy()
    ima.to_zero()
    for j in range(numref):
        #  even, odd, numer of even, number of images.  After frc, totav
        refi.append([get_im(refim,j), ima.copy(), 0])
        refi[j][0].process_inplace("normalize.mask", {"mask":mask, "no_sigma":1}) # normalize reference images to N(0,1)

    # create/reset alignment parameters
    for im in range(nima):
        #data[im].set_attr( 'ID', list_of_particles[im] )
        util.set_params2D( data[im], [0.0, 0.0, 0.0, 0, 1.0], 'xform.align2d' )
        data[im].process_inplace("normalize.mask", {"mask":mask, "no_sigma":0}) # subtract average under the mask
        #data[im] -= st[0]
        if phase_flip:
            data[im] = filt_ctf(data[im], data[im].get_attr("ctf"), binary = True)

    # precalculate rings & weights
    numr = alignment.Numrinit( first_ring, last_ring, rstep, mode )
    wr   = alignment.ringwe( numr, mode )
    RangePop()
    if myid == main_node:  
        seed(rand_seed)
        a0 = -1.0
        ref_data = [mask, center, None, None]

    again = True
    Iter = 0
    
    
    #----------------------------------[ gpu setup ]

    print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "mref_ali2d_gpu() :: MPI proc["+str(myid)+"] run pre-alignment GPU setup" )

    # alignment parameters
    aln_cfg = AlignConfig(
        len(data), numref,                # no. of images & averages
        data[0].get_xsize(),         # image size (images are always assumed to be square)
        numr[-3], 256,               # polar sampling params (no. of rings, no. of sample on ring)
        step[0], xrng[0], xrng[0])   # shift params (step size & shift range in x/y dim)
    aln_cfg.freeze()

    # find largest batch size we can fit on the given card
    gpu_batch_limit = 0
    RangePush("Determine batch size")
    for split in [ 2**i for i in range(int(math.log(len(data),2))+1) ][::-1]:
        aln_cfg.sbj_num = min( gpu_batch_limit + split, len(data) )
        if cu_module.pre_align_size_check( ctypes.c_uint(len(data)), ctypes.byref(aln_cfg), ctypes.c_uint(myid), ctypes.c_float(cuda_device_occ), ctypes.c_bool(False) ) == True:
            gpu_batch_limit += split

    gpu_batch_limit = aln_cfg.sbj_num
    RangePop()
    
    print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "mref_ali2d_gpu() :: GPU["+str(myid)+"] pre-alignment batch size: %d/%d" % (gpu_batch_limit, len(data)) )

    # initialize gpu resources (returns location for our alignment parameters in CUDA unified memory)
    gpu_aln_param = cu_module.pre_align_init( ctypes.c_uint(len(data)), ctypes.byref(aln_cfg), ctypes.c_uint(myid) )
    gpu_aln_param = ctypes.cast( gpu_aln_param, aln_param_ptr ) # cast to allow Python-side r/w access

    gpu_batch_count = len(data)/gpu_batch_limit if len(data)%gpu_batch_limit==0 else len(data)//gpu_batch_limit+1

    print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "mref_ali2d_gpu() :: GPU["+str(myid)+"] batch count:", gpu_batch_count )

    # if the local stack fits on the gpu we only fetch the img data and the reference data once before we loop
    if( gpu_batch_count == 1 ):
        cu_module.pre_align_fetch(
            get_c_ptr_array(data),
            ctypes.c_uint(len(data)), 
            ctypes.c_char_p("sbj_batch") )
            
    #----------------------------------[ alignment ]

    print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "mref_ali2d_gpu() :: MPI proc["+str(myid)+"] start alignment iterations" )

    N_step = 0
    # set gpu search parameters
    cu_module.reset_shifts( ctypes.c_float(xrng[N_step]), ctypes.c_float(step[N_step]) )
        

    
    while Iter < max_iter and again:
        
        cu_module.pre_align_fetch(
            get_c_ptr_array([tmp[0] for tmp in refi]),
            ctypes.c_int(numref), 
            ctypes.c_char_p("ref_batch"))
        
        # backup last iteration's alignment parameters
        RangePush("Iteration's alignment parameters")
        old_ali_params = []
        for im in range(nima):  
            alpha, sx, sy, mirror, scale = util.get_params2D(data[im])
            old_ali_params.extend([alpha, sx, sy, mirror])
        RangePop()  
        
        ############################################GPU
        #
        # FOR gpu_batch_i DO ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ >
        RangePush("run the alignment on gpu")
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
            cu_module.mref_align_run( ctypes.c_int(gpu_batch_start), ctypes.c_int(gpu_batch_end))
            
            # print progress bar
            gpu_calls_ttl = 1 * max_iter * gpu_batch_count
            #gpu_calls_ttl = len(xrng) * max_iter * gpu_batch_count - 1
            gpu_calls_cnt = N_step*max_iter*gpu_batch_count + Iter*gpu_batch_count + gpu_batch_idx
            gpu_calls_prc = int( float(gpu_calls_cnt+1)/gpu_calls_ttl * 50.0 )
            sys.stdout.write( "\r[MREF-ALIGN][GPU"+str(myid)+"][" + "="*gpu_calls_prc + "-"*(50-gpu_calls_prc) + "]~[%d/%d]~[%.2f%%]\n" % (gpu_calls_cnt+1, gpu_calls_ttl, (float(gpu_calls_cnt+1)/gpu_calls_ttl)*100.0) )
            sys.stdout.flush()
            if gpu_calls_cnt+1 == gpu_calls_ttl: print("")
        RangePop() 
        # < ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ FOR gpu_batch_i DO
        #set reference image to zero
        RangePush("reference image to zero")
        for j in range(numref):
            refi[j][0].to_zero()
            refi[j][1].to_zero()
            refi[j][2] = 0
        RangePop()
        
        RangePush("transfer angle and average")
        assign = [[] for i in range(numref)]
        # ------------------------------------------------------------
        gpu_image_ptr = cu_module.get_gpu_img_ptr()
        print("type of gpu aln pointer = ",type(gpu_aln_param))
        print("="*30)
        print("type of gpu img pointer = ",type(gpu_image_ptr))
        print("="*30)
        # ------------------------------------------------------------
        #begin MPI section
        for k, img in enumerate(data):
            # this is usually done in ormq()
            angle   =  gpu_aln_param[k].angle
            sx_neg  = -gpu_aln_param[k].shift_x
            sy_neg  = -gpu_aln_param[k].shift_y
            c_ang   =  math.cos( math.radians(angle) )
            s_ang   = -math.sin( math.radians(angle) )
            shift_x =  sx_neg*c_ang - sy_neg*s_ang
            shift_y =  sx_neg*s_ang + sy_neg*c_ang
            mn = gpu_aln_param[k].mirror
            iref = int(gpu_aln_param[k].ref_id)
            # this happens in ali2d_single_iter()
            set_params2D( img, [angle, shift_x, shift_y, int(mn), 1.0], "xform.align2d" )
            img.set_attr('assign',iref)
            #apply current parameters and add to the average
            temp = rot_shift2D(img,angle,shift_x,shift_y,mn)
            it = k%2
            Util.add_img(refi[iref][it],temp)
            assign[iref].append(list_of_particles[k])
            refi[iref][2] += 1.0
            #print("Assign to %d reference for image %d"%(iref,list_of_particles[k]))
        RangePop()
            
            
        RangePush("average")
        for j in range(numref):
            reduce_EMData_to_root(refi[j][0],myid,main_node,comm = mpi_comm)
            reduce_EMData_to_root(refi[j][1],myid,main_node,comm = mpi_comm)
            refi[j][2] = mpi_reduce(refi[j][2], 1, MPI_FLOAT, MPI_SUM, main_node, mpi_comm)
            if(myid == main_node): refi[j][2] = int(refi[j][2][0])          
        RangePop()
        
        RangePush("set param")
        #gather assignments
        for j in range(numref):
            if myid == main_node:
                for n in range(number_of_proc):
                    if n != main_node:
                        import sp_global_def
                        ln =  mpi_recv(1, MPI_INT, n, sp_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi_comm)
                        lis = mpi_recv(ln[0], MPI_INT, n, sp_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi_comm)
                        for l in range(ln[0]): assign[j].append(int(lis[l]))
            else:
                import sp_global_def
                mpi_send(len(assign[j]), 1, MPI_INT, main_node, sp_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi_comm)
                mpi_send(assign[j], len(assign[j]), MPI_INT, main_node, sp_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi_comm)
        
        if myid == main_node:
            # replace the name of the stack with reference with the current one
            refim = os.path.join(outdir,"aqm%03d.hdf"%Iter)
            a1 = 0.0
            ave_fsc = []
            for j in range(numref):
                if refi[j][2] < 4:
                    #ERROR("One of the references vanished","mref_ali2d_MPI",1)
                    #  if vanished, put a random image (only from main node!) there
                    assign[j] = []
                    assign[j].append(randint(0,nima-1))
                    refi[j][0] = data[assign[j][0]].copy()

                else:
                    #frsc = fsc_mask(refi[j][0], refi[j][1], mask, 1.0, os.path.join(outdir,"drm%03d%04d"%(Iter, j)))
                    from sp_statistics import fsc
                    frsc = fsc(refi[j][0], refi[j][1], 1.0, os.path.join(outdir,"drm%03d%04d.txt"%(Iter,j)))
                    Util.add_img( refi[j][0], refi[j][1] )
                    Util.mul_scalar( refi[j][0], 1.0/float(refi[j][2]) )
                            
                    if ave_fsc == []:
                        for i in range(len(frsc[1])): ave_fsc.append(frsc[1][i])
                        c_fsc = 1
                    else:
                        for i in range(len(frsc[1])): ave_fsc[i] += frsc[1][i]
                        c_fsc += 1
                    #print 'OK', j, len(frsc[1]), frsc[1][0:5], ave_fsc[0:5]            
        
            #print 'sum', sum(ave_fsc)
            if sum(ave_fsc) != 0:        
                for i in range(len(ave_fsc)):
                    ave_fsc[i] /= float(c_fsc)
                    frsc[1][i]  = ave_fsc[i]
            
            for j in range(numref):
                ref_data[2]    = refi[j][0]
                ref_data[3]    = frsc
                
                refi[j][0], cs = user_func(ref_data)    
                
                # write the current average
                TMP = []
                for i_tmp in range(len(assign[j])): TMP.append(float(assign[j][i_tmp]))
                TMP.sort()
                refi[j][0].set_attr_dict({'ave_n': refi[j][2],  'members': TMP })
                del TMP
                refi[j][0].process_inplace("normalize.mask", {"mask":mask, "no_sigma":1})
                refi[j][0].write_image(refim, j)
                
            Iter += 1
            msg = "ITERATION #%3d        %d\n\n"%(Iter,again)
            print_msg(msg)
            for j in range(numref):
                msg = "   group #%3d   number of particles = %7d\n"%(j, refi[j][2])
                print_msg(msg)
        Iter  = bcast_number_to_all(Iter, main_node, mpi_comm) # need to tell all
        if again:
            for j in range(numref):
                bcast_EMData_to_all(refi[j][0], myid, main_node, mpi_comm)
        RangePop()
    # clean up 
    del assign
    RangePush("disk")
    # write out headers  and STOP, under MPI writing has to be done sequentially (time-consumming)
    mpi_barrier(mpi_comm)
    if CTF and data_had_ctf == 0:
        for im in range(nima): data[im].set_attr('ctf_applied', 0)
    par_str = ['xform.align2d', 'assign', 'ID']
    if myid == main_node:
        from sp_utilities import file_type
        if(file_type(filename) == "bdb"):
            from sp_utilities import recv_attr_dict_bdb
            recv_attr_dict_bdb(main_node, filename, data, par_str, image_start, image_end, number_of_proc)
        else:
            from sp_utilities import recv_attr_dict
            recv_attr_dict(main_node, filename, data, par_str, image_start, image_end, number_of_proc)
    else:           send_attr_dict(main_node, data, par_str, image_start, image_end)
        
    # free gpu resources
    cu_module.gpu_clear()
    mpi.mpi_barrier(mpi_comm)
    RangePop()
    
    if myid == main_node:
        print_end_msg("mref_ali2d_GPU")






def mref_ali2d_MPI(stack, refim, outdir, maskfile = None, ir=1, ou=-1, rs=1, xrng=0, yrng=0, step=1, center=1, maxit=10, CTF=False, snr=1.0, user_func_name="ref_ali2d", rand_seed=1000):
    # 2D multi-reference alignment using rotational ccf in polar coordinates and quadratic interpolation

    from sp_utilities      import   model_circle, combine_params2, inverse_transform2, drop_image, get_image, get_im
    from sp_utilities      import   reduce_EMData_to_root, bcast_EMData_to_all, bcast_number_to_all
    from sp_utilities      import   send_attr_dict
    from sp_utilities        import   center_2D
    from sp_statistics     import   fsc_mask
    from sp_alignment      import   Numrinit, ringwe, search_range
    from sp_fundamentals   import   rot_shift2D, fshift
    from sp_utilities      import   get_params2D, set_params2D
    from random         import   seed, randint
    from sp_morphology     import   ctf_2
    from sp_filter         import   filt_btwl, filt_params
    from numpy          import   reshape, shape
    from sp_utilities      import   print_msg, print_begin_msg, print_end_msg
    import os
    import sys
    import shutil
    from sp_applications   import MPI_start_end
    from mpi       import mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD
    from mpi       import mpi_reduce, mpi_bcast, mpi_barrier, mpi_recv, mpi_send
    from mpi       import MPI_SUM, MPI_FLOAT, MPI_INT
    

    number_of_proc = mpi_comm_size(MPI_COMM_WORLD)
    myid = mpi_comm_rank(MPI_COMM_WORLD)
    main_node = 0
    
    # create the output directory, if it does not exist

    if os.path.exists(outdir):  ERROR('Output directory exists, please change the name and restart the program', "mref_ali2d_MPI ", 1, myid)
    mpi_barrier(MPI_COMM_WORLD)

    import sp_global_def
    if myid == main_node:
        os.mkdir(outdir)
        sp_global_def.LOGFILE =  os.path.join(outdir, sp_global_def.LOGFILE)
        print_begin_msg("mref_ali2d_MPI")

    nima = EMUtil.get_image_count(stack)
    
    image_start, image_end = MPI_start_end(nima, number_of_proc, myid)

    nima = EMUtil.get_image_count(stack)
    ima = EMData()
    ima.read_image(stack, image_start)

    first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit)

    if max_iter == 0:
        max_iter  = 10
        auto_stop = True
    else:
        auto_stop = False

    if myid == main_node:
        print_msg("Input stack                 : %s\n"%(stack))
        print_msg("Reference stack             : %s\n"%(refim))    
        print_msg("Output directory            : %s\n"%(outdir))
        print_msg("Maskfile                    : %s\n"%(maskfile))
        print_msg("Inner radius                : %i\n"%(first_ring))

    nx = ima.get_xsize()
    # default value for the last ring
    if last_ring == -1: last_ring=nx/2-2
    
    if myid == main_node:
        print_msg("Outer radius                : %i\n"%(last_ring))
        print_msg("Ring step                   : %i\n"%(rstep))
        print_msg("X search range              : %f\n"%(xrng))
        print_msg("Y search range              : %f\n"%(yrng))
        print_msg("Translational step          : %f\n"%(step))
        print_msg("Center type                 : %i\n"%(center))
        print_msg("Maximum iteration           : %i\n"%(max_iter))
        print_msg("CTF correction              : %s\n"%(CTF))
        print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
        print_msg("Random seed                 : %i\n\n"%(rand_seed))    
        print_msg("User function               : %s\n"%(user_func_name))
    import sp_user_functions
    user_func = sp_user_functions.factory[user_func_name]

    if maskfile:
        import  types
        if type(maskfile) is bytes:  mask = get_image(maskfile)
        else: mask = maskfile
    else : mask = model_circle(last_ring, nx, nx)
    #  references, do them on all processors...
    refi = []
    numref = EMUtil.get_image_count(refim)

    # IMAGES ARE SQUARES! center is in SPIDER convention
    cnx = nx/2+1
    cny = cnx

    mode = "F"
    #precalculate rings
    numr = Numrinit(first_ring, last_ring, rstep, mode)
    wr = ringwe(numr, mode)
    
    
    # prepare reference images on all nodes
    ima.to_zero()
    for j in range(numref):
        #  even, odd, number of even, number of images.  After frc, totav
        refi.append([get_im(refim,j), ima.copy(), 0])
    #  for each node read its share of data
    data = EMData.read_images(stack, list(range(image_start, image_end)))
    for im in range(image_start, image_end):
        data[im-image_start].set_attr('ID', im)

    if myid == main_node:  seed(rand_seed)

    a0 = -1.0
    again = True
    Iter = 0

    ref_data = [mask, center, None, None]

    while Iter < max_iter and again:
        ringref = []
        mashi = cnx-last_ring-2
        for j in range(numref):
            refi[j][0].process_inplace("normalize.mask", {"mask":mask, "no_sigma":1}) # normalize reference images to N(0,1)
            cimage = Util.Polar2Dm(refi[j][0] , cnx, cny, numr, mode)
            Util.Frngs(cimage, numr)
            Util.Applyws(cimage, numr, wr)
            ringref.append(cimage)
            # zero refi
            refi[j][0].to_zero()
            refi[j][1].to_zero()
            refi[j][2] = 0
        
        assign = [[] for i in range(numref)]
        # begin MPI section
        for im in range(image_start, image_end):
            alpha, sx, sy, mirror, scale = get_params2D(data[im-image_start])
            #  Why inverse?  07/11/2015 PAP
            alphai, sxi, syi, scalei = inverse_transform2(alpha, sx, sy)
            # normalize
            data[im-image_start].process_inplace("normalize.mask", {"mask":mask, "no_sigma":0}) # subtract average under the mask
            # If shifts are outside of the permissible range, reset them
            if(abs(sxi)>mashi or abs(syi)>mashi):
                sxi = 0.0
                syi = 0.0
                set_params2D(data[im-image_start],[0.0,0.0,0.0,0,1.0])
            ny = nx
            txrng = search_range(nx, last_ring, sxi, xrng, "mref_ali2d_MPI")
            txrng = [txrng[1],txrng[0]]
            tyrng = search_range(ny, last_ring, syi, yrng, "mref_ali2d_MPI")
            tyrng = [tyrng[1],tyrng[0]]
            # align current image to the reference
            [angt, sxst, syst, mirrort, xiref, peakt] = Util.multiref_polar_ali_2d(data[im-image_start], 
                ringref, txrng, tyrng, step, mode, numr, cnx+sxi, cny+syi)
            
        
            iref = int(xiref)
            # combine parameters and set them to the header, ignore previous angle and mirror
            [alphan, sxn, syn, mn] = combine_params2(0.0, -sxi, -syi, 0, angt, sxst, syst, (int)(mirrort))
            set_params2D(data[im-image_start], [alphan, sxn, syn, int(mn), scale])
            data[im-image_start].set_attr('assign',iref)
            # apply current parameters and add to the average
            temp = rot_shift2D(data[im-image_start], alphan, sxn, syn, mn)
            it = im%2
            Util.add_img( refi[iref][it], temp)
            assign[iref].append(im)
            #assign[im] = iref
            refi[iref][2] += 1.0
        del ringref
        # end MPI section, bring partial things together, calculate new reference images, broadcast them back
        
        for j in range(numref):
            reduce_EMData_to_root(refi[j][0], myid, main_node)
            reduce_EMData_to_root(refi[j][1], myid, main_node)
            refi[j][2] = mpi_reduce(refi[j][2], 1, MPI_FLOAT, MPI_SUM, main_node, MPI_COMM_WORLD)
            if(myid == main_node): refi[j][2] = int(refi[j][2][0])
        # gather assignements
        for j in range(numref):
            if myid == main_node:
                for n in range(number_of_proc):
                    if n != main_node:
                        import sp_global_def
                        ln =  mpi_recv(1, MPI_INT, n, sp_global_def.SPARX_MPI_TAG_UNIVERSAL, MPI_COMM_WORLD)
                        lis = mpi_recv(ln[0], MPI_INT, n, sp_global_def.SPARX_MPI_TAG_UNIVERSAL, MPI_COMM_WORLD)
                        for l in range(ln[0]): assign[j].append(int(lis[l]))
            else:
                import sp_global_def
                mpi_send(len(assign[j]), 1, MPI_INT, main_node, sp_global_def.SPARX_MPI_TAG_UNIVERSAL, MPI_COMM_WORLD)
                mpi_send(assign[j], len(assign[j]), MPI_INT, main_node, sp_global_def.SPARX_MPI_TAG_UNIVERSAL, MPI_COMM_WORLD)
        
        if myid == main_node:
            # replace the name of the stack with reference with the current one
            refim = os.path.join(outdir,"aqm%03d.hdf"%Iter)
            a1 = 0.0
            ave_fsc = []
            for j in range(numref):
                if refi[j][2] < 4:
                    #ERROR("One of the references vanished","mref_ali2d_MPI",1)
                    #  if vanished, put a random image (only from main node!) there
                    assign[j] = []
                    assign[j].append( randint(image_start, image_end-1) - image_start )
                    refi[j][0] = data[assign[j][0]].copy()
                    #print 'ERROR', j
                else:
                    #frsc = fsc_mask(refi[j][0], refi[j][1], mask, 1.0, os.path.join(outdir,"drm%03d%04d"%(Iter, j)))
                    from sp_statistics import fsc
                    frsc = fsc(refi[j][0], refi[j][1], 1.0, os.path.join(outdir,"drm%03d%04d.txt"%(Iter,j)))
                    Util.add_img( refi[j][0], refi[j][1] )
                    Util.mul_scalar( refi[j][0], 1.0/float(refi[j][2]) )
                            
                    if ave_fsc == []:
                        for i in range(len(frsc[1])): ave_fsc.append(frsc[1][i])
                        c_fsc = 1
                    else:
                        for i in range(len(frsc[1])): ave_fsc[i] += frsc[1][i]
                        c_fsc += 1
                    #print 'OK', j, len(frsc[1]), frsc[1][0:5], ave_fsc[0:5]            
        
            
            #print 'sum', sum(ave_fsc)
            if sum(ave_fsc) != 0:        
                for i in range(len(ave_fsc)):
                    ave_fsc[i] /= float(c_fsc)
                    frsc[1][i]  = ave_fsc[i]
            
            for j in range(numref):
                ref_data[2]    = refi[j][0]
                ref_data[3]    = frsc
                refi[j][0], cs = user_func(ref_data)    
        
                # write the current average
                TMP = []
                for i_tmp in range(len(assign[j])): TMP.append(float(assign[j][i_tmp]))
                TMP.sort()
                refi[j][0].set_attr_dict({'ave_n': refi[j][2],  'members': TMP })
                del TMP
                refi[j][0].process_inplace("normalize.mask", {"mask":mask, "no_sigma":1})
                refi[j][0].write_image(refim, j)
                
            Iter += 1
            msg = "ITERATION #%3d        %d\n\n"%(Iter,again)
            print_msg(msg)
            for j in range(numref):
                msg = "   group #%3d   number of particles = %7d\n"%(j, refi[j][2])
                print_msg(msg)
        Iter  = bcast_number_to_all(Iter, main_node) # need to tell all
        if again:
            for j in range(numref):
                bcast_EMData_to_all(refi[j][0], myid, main_node)

    #  clean up
    del assign
    # write out headers  and STOP, under MPI writing has to be done sequentially (time-consumming)
    mpi_barrier(MPI_COMM_WORLD)
    if CTF and data_had_ctf == 0:
        for im in range(len(data)): data[im].set_attr('ctf_applied', 0)
    par_str = ['xform.align2d', 'assign', 'ID']
    if myid == main_node:
        from sp_utilities import file_type
        if(file_type(stack) == "bdb"):
            from sp_utilities import recv_attr_dict_bdb
            recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
        else:
            from sp_utilities import recv_attr_dict
            recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
    else:           send_attr_dict(main_node, data, par_str, image_start, image_end)
    if myid == main_node:
        print_end_msg("mref_ali2d_GPU")

def mref_ali2d(stack, refim, outdir, maskfile=None, ir=1, ou=-1, rs=1, xrng=0, yrng=0, step=1, center=1, maxit=0, CTF=False, snr=1.0, user_func_name="ref_ali2d", rand_seed=1000, MPI=False):
    """
        Name
            mref_ali2d - Perform 2-D multi-reference alignment of an image series
        Input
            stack: set of 2-D images in a stack file, images have to be squares
            refim: set of initial reference 2-D images in a stack file 
            maskfile: optional maskfile to be used in the alignment
            inner_radius: inner radius for rotational correlation > 0
            outer_radius: outer radius for rotational correlation < nx/2-1
            ring_step: step between rings in rotational correlation >0
            x_range: range for translation search in x direction, search is +/xr 
            y_range: range for translation search in y direction, search is +/yr 
            translation_step: step of translation search in both directions
            center: center the average
            max_iter: maximum number of iterations the program will perform
            CTF: if this flag is set, the program will use CTF information provided in file headers
            snr: signal-to-noise ratio of the data
            rand_seed: the seed used for generating random numbers
            MPI: whether to use MPI version
        Output
            output_directory: directory name into which the output files will be written.
            header: the alignment parameters are stored in the headers of input files as 'xform.align2d'.
    """
# 2D multi-reference alignment using rotational ccf in polar coordinates and quadratic interpolation
    if MPI:
        mref_ali2d_MPI(stack, refim, outdir, maskfile, ir, ou, rs, xrng, yrng, step, center, maxit, CTF, snr, user_func_name, rand_seed)
        return

    from sp_utilities      import   model_circle, combine_params2, inverse_transform2, drop_image, get_image
    from sp_utilities        import   center_2D, get_im, get_params2D, set_params2D
    from sp_statistics     import   fsc
    from sp_alignment      import   Numrinit, ringwe, fine_2D_refinement, search_range
    from sp_fundamentals   import   rot_shift2D, fshift
    from random         import   seed, randint
    import os
    import sys
    
    from sp_utilities      import   print_begin_msg, print_end_msg, print_msg
    import shutil


    
    # create the output directory, if it does not exist
    if os.path.exists(outdir):  shutil.rmtree(outdir) #ERROR('Output directory exists, please change the name and restart the program', "mref_ali2d", 1)
    os.mkdir(outdir)
    import sp_global_def
    sp_global_def.LOGFILE =  os.path.join(outdir, sp_global_def.LOGFILE)
    
    first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit)

    print_begin_msg("mref_ali2d")

    print_msg("Input stack                 : %s\n"%(stack))
    print_msg("Reference stack             : %s\n"%(refim))    
    print_msg("Output directory            : %s\n"%(outdir))
    print_msg("Maskfile                    : %s\n"%(maskfile))
    print_msg("Inner radius                : %i\n"%(first_ring))

    ima = EMData()
    ima.read_image(stack, 0)
    nx = ima.get_xsize()
    # default value for the last ring
    if last_ring == -1: last_ring = nx/2-2

    print_msg("Outer radius                : %i\n"%(last_ring))
    print_msg("Ring step                   : %i\n"%(rstep))
    print_msg("X search range              : %i\n"%(xrng))
    print_msg("Y search range              : %i\n"%(yrng))
    print_msg("Translational step          : %i\n"%(step))
    print_msg("Center type                 : %i\n"%(center))
    print_msg("Maximum iteration           : %i\n"%(max_iter))
    print_msg("CTF correction              : %s\n"%(CTF))
    print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
    print_msg("Random seed                 : %i\n\n"%(rand_seed))
    print_msg("User function               : %s\n"%(user_func_name))
    output = sys.stdout

    import sp_user_functions
    user_func = sp_user_functions.factory[user_func_name]

    if maskfile:
        import types
        if type(maskfile) is bytes:  mask = get_image(maskfile)
        else: mask = maskfile
    else: mask = model_circle(last_ring, nx, nx)
    #  references
    refi = []
    numref = EMUtil.get_image_count(refim)
        
    # IMAGES ARE SQUARES! center is in SPIDER convention
    cnx = nx/2+1
    cny = cnx

    mode = "F"
    #precalculate rings
    numr = Numrinit(first_ring, last_ring, rstep, mode)
    wr = ringwe(numr, mode)
    # reference images
    params = []
    #read all data
    data = EMData.read_images(stack)
    nima = len(data)
    # prepare the reference
    ima.to_zero()
    for j in range(numref):
        temp = EMData()
        temp.read_image(refim, j)
        #  eve, odd, numer of even, number of images.  After frc, totav
        refi.append([temp, ima.copy(), 0])

    seed(rand_seed)
    again = True
    
    ref_data = [mask, center, None, None]

    Iter = 0
    
    
    while Iter < max_iter and again:
        ringref = []
        #print "numref",numref
        
        ### Reference ###
        mashi = cnx-last_ring-2
        for j in range(numref):
            refi[j][0].process_inplace("normalize.mask", {"mask":mask, "no_sigma":1})
            cimage = Util.Polar2Dm(refi[j][0], cnx, cny, numr, mode)
            Util.Frngs(cimage, numr)
            Util.Applyws(cimage, numr, wr)
            ringref.append(cimage)
        
        assign = [[] for i in range(numref)]
        sx_sum = [0.0]*numref
        sy_sum = [0.0]*numref
        for im in range(nima):
            alpha, sx, sy, mirror, scale = get_params2D(data[im])
            #  Why inverse?  07/11/2015  PAP
            alphai, sxi, syi, scalei = inverse_transform2(alpha, sx, sy)
            # normalize
            data[im].process_inplace("normalize.mask", {"mask":mask, "no_sigma":0})
            # If shifts are outside of the permissible range, reset them
            if(abs(sxi)>mashi or abs(syi)>mashi):
                sxi = 0.0
                syi = 0.0
                set_params2D(data[im],[0.0,0.0,0.0,0,1.0])
            ny = nx
            txrng = search_range(nx, last_ring, sxi, xrng, "mref_ali2d")
            txrng = [txrng[1],txrng[0]]
            tyrng = search_range(ny, last_ring, syi, yrng, "mref_ali2d")
            tyrng = [tyrng[1],tyrng[0]]
            # align current image to the reference
            #[angt, sxst, syst, mirrort, xiref, peakt] = Util.multiref_polar_ali_2d_p(data[im], 
            #    ringref, txrng, tyrng, step, mode, numr, cnx+sxi, cny+syi)
            #print(angt, sxst, syst, mirrort, xiref, peakt)
            [angt, sxst, syst, mirrort, xiref, peakt] = Util.multiref_polar_ali_2d(data[im], 
                ringref, txrng, tyrng, step, mode, numr, cnx+sxi, cny+syi)
            
            iref = int(xiref)
            # combine parameters and set them to the header, ignore previous angle and mirror
            [alphan, sxn, syn, mn] = combine_params2(0.0, -sxi, -syi, 0, angt, sxst, syst, int(mirrort))
            set_params2D(data[im], [alphan, sxn, syn, int(mn), scale])
            if mn == 0: sx_sum[iref] += sxn
            else: sx_sum[iref] -= sxn
            sy_sum[iref] += syn
            data[im].set_attr('assign', iref)
            # apply current parameters and add to the average
            temp = rot_shift2D(data[im], alphan, sxn, syn, mn)
            it = im%2
            Util.add_img(refi[iref][it], temp)
        
            assign[iref].append(im)
            refi[iref][2] += 1
        del ringref
        if again:
            a1 = 0.0
            for j in range(numref):
                msg = "   group #%3d   number of particles = %7d\n"%(j, refi[j][2])
                print_msg(msg)
                if refi[j][2] < 4:
                    #ERROR("One of the references vanished","mref_ali2d",1)
                    #  if vanished, put a random image there
                    assign[j] = []
                    assign[j].append(randint(0, nima-1))
                    refi[j][0] = data[assign[j][0]].copy()
                else:
                    max_inter = 0  # switch off fine refi.
                    br = 1.75
                    #  the loop has to 
                    for INter in range(max_inter+1):
                        # Calculate averages at least ones, meaning even if no within group refinement was requested
                        frsc = fsc(refi[j][0], refi[j][1], 1.0, os.path.join(outdir,"drm_%03d_%04d.txt"%(Iter, j)))
                        Util.add_img(refi[j][0], refi[j][1])
                        Util.mul_scalar(refi[j][0], 1.0/float(refi[j][2]))
                            
                        ref_data[2] = refi[j][0]
                        ref_data[3] = frsc                        
                        refi[j][0], cs = user_func(ref_data)
                        if center == -1:
                            cs[0] = sx_sum[j]/len(assign[j])
                            cs[1] = sy_sum[j]/len(assign[j])
                            refi[j][0] = fshift(refi[j][0], -cs[0], -cs[1])
                        for i in range(len(assign[j])):
                            im = assign[j][i]
                            alpha, sx, sy, mirror, scale =  get_params2D(data[im])
                            alphan, sxn, syn, mirrorn = combine_params2(alpha, sx, sy, mirror, 0.0, -cs[0], -cs[1], 0)
                            set_params2D(data[im], [alphan, sxn, syn, int(mirrorn), scale])
                        # refine images within the group
                        #  Do the refinement only if max_inter>0, but skip it for the last iteration.
                        if INter < max_inter:
                            fine_2D_refinement(data, br, mask, refi[j][0], j)
                            #  Calculate updated average
                            refi[j][0].to_zero()
                            refi[j][1].to_zero()
                            for i in range(len(assign[j])):
                                im = assign[j][i]
                                alpha, sx, sy, mirror, scale = get_params2D(data[im])
                                # apply current parameters and add to the average
                                temp = rot_shift2D(data[im], alpha, sx, sy, mn)
                                it = im%2
                                Util.add_img(refi[j][it], temp)  
                # write the current average
                TMP = []
                for i_tmp in range(len(assign[j])):  TMP.append(float(assign[j][i_tmp]))
                TMP.sort()
                refi[j][0].set_attr_dict({'ave_n': refi[j][2], 'members': TMP })
                del TMP
                # replace the name of the stack with reference with the current one
                newrefim = os.path.join(outdir,"aqm%03d.hdf"%Iter)
                refi[j][0].write_image(newrefim, j)
            Iter += 1
            msg = "ITERATION #%3d        \n"%(Iter)
            print_msg(msg)
            
    newrefim = os.path.join(outdir,"multi_ref.hdf")
    for j in range(numref):  refi[j][0].write_image(newrefim, j)
    from sp_utilities import write_headers
    write_headers(stack, data, list(range(nima)))        
    print_end_msg("mref_ali2d")

def mpi_assert( condition, msg ):
    if not condition:
        mpi_rank = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
        print( "MPI PROC["+str(mpi_rank)+"] ASSERTION ERROR:", msg, file=sys.stderr)
        sys.stderr.flush()
        mpi.mpi_finalize()
        sys.exit()

def main():
    arglist = []
    for arg in sys.argv:
        arglist.append( arg )
    progname = os.path.basename(sys.argv[0])
    usage = progname + " data_stack reference_stack outdir <maskfile> --ir=inner_radius --ou=outer_radius --rs=ring_step --xr=x_range --yr=y_range  --ts=translation_step --center=center_type --maxit=max_iteration --CTF --snr=SNR --function=user_function_name --rand_seed=random_seed --CUDA --GPUID --MPI"
    parser = OptionParser(usage,version=SPARXVERSION)
    parser.add_option("--ir", type="float", default=1, help="  inner radius for rotational correlation > 0 (set to 1)")
    parser.add_option("--ou", type="float", default=-1, help="  outer radius for rotational correlation < nx/2-1 (set to the radius of the particle)")
    parser.add_option("--rs", type="float", default=1, help="  step between rings in rotational correlation > 0 (set to 1)" )
    parser.add_option("--xr", type="float", default=0, help="  range for translation search in x direction, search is +/-xr ")
    parser.add_option("--yr", type="float", default=0, help="  range for translation search in y direction, search is +/-yr ")
    parser.add_option("--ts", type="float", default=1, help="  step of translation search in both directions")
    parser.add_option("--center", type="float", default=1, help="  0 - if you do not want the average to be centered, 1 - center the average (default=1)")
    parser.add_option("--maxit", type="float", default=10, help="  maximum number of iterations (set to 10) ")
    parser.add_option("--CTF", action="store_true", default=False, help=" Consider CTF correction during multiple reference alignment")
    parser.add_option("--snr", type="float",  default= 1.0, help="  signal-to-noise ratio of the data (set to 1.0)")
    parser.add_option("--function", type="string", default="ref_ali2d", help="  name of the reference preparation function")
    parser.add_option("--rand_seed", type="int", default=1000, help=" random seed of initial (set to 1000)" )
    parser.add_option("--gpu_devices",     type="string",       default="",    help="Specify the GPUs to be used (e.g. --gpu_devices=0, or --gpu_devices=0,1 for one or two GPUs, respectively). Using \"$ nividia-smi\" in the terminal will print out what GPUs are available. [Default: None]" )
    parser.add_option( "--gpu_info",         action="store_true", default=False, help="Print detailed information about the selected GPUs. Use --gpu_devices to specify what GPUs you want to know about. NOTE: program will stop after printing this information, so don't use this parameter if you intend to actually process any data. [Default: False]" )
    parser.add_option("--MPI", action="store_true", default=False,     help="  whether to use MPI version ")
    parser.add_option("--EQ", action="store_true", default=False,     help="  equal version ")
    (options, args) = parser.parse_args(arglist[1:])
    if len(args) < 3 or len(args) > 4:
            print("usage: " + usage)
            print("Please run '" + progname + " -h' for detailed options")
    else:
        
        if args[2] == 'None':
            outdir = None
        else:
            outdir = args[2]

        if len(args) == 3:
            mask = None
        else:
            mask = args[3]

        if global_def.CACHE_DISABLE:
            from utilities import disable_bdb_cache
            disable_bdb_cache()

        prefix = os.path.join(outdir)
        prefix += "/"
        global_def.LOGFILE = prefix + global_def.LOGFILE
        print(global_def.LOGFILE)

        global_def.BATCH = True
        if options.MPI:
            from mpi import mpi_init, mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD
            import mpi
            print("use mpi")
            sys.argv = mpi_init(len(sys.argv), sys.argv) # init and finalize are needed for each process

            mref_ali2d_MPI(args[0], args[1], args[2], mask, options.ir, options.ou, options.rs, options.xr, options.yr, options.ts, options.center, options.maxit, options.CTF, options.snr, options.function, options.rand_seed)
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
            # 
            # mpi_comm_split_type(comm, split_type, key, info)
            # partitions the group associated with comm into disjoint subgroups, based on the type specied by split_type. 
            # Each subgroup contains all processes of the same type. Within each subgroup, the processes are ranked in the 
            # order defined by the value of the argument key, with ties broken according to their rank in the old group.
            # 
            # Args:
            #     comm      : Communicator
            #     split_type: Type of processes to be grouped together
            #     key       : Control of rank assignment
            #     info      : Info argument 
            # 
            # Returns:
            #     newcomm   : New communicator to handle subgroups
            # 
            # Notes:
            #     split_type=MPI_COMM_TYPE_SHARED: splits the communicator into subcommunicators, 
            #                                       each of which can create a shared memory region.
            #     key=0                          : preserve the old rank order as the new rank order
            #     info=MPI_INFO_NULL             : no info
            # 
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
            
            # mpi_comm_split(comm, color, key) 
            # partitions the group associated with comm into disjoint subgroups, one for each value of color.
            # Each subgroup contains all processes of the same color. Within each subgroup, the processes are 
            # ranked in the order defined by the value of the argument key, with ties broken according to their 
            # rank in the old group.
            # 
            # Args:
            #     comm : Communicator
            #     color: Control of subset assignment
            #     keys : Control of rank assignment 
            # 
            # Returns:
            #     newcomm: New communicator to handle subgroups
            # 
            # ex:
            #     in this function, the original world MPI will be divided into two subgroups, one is GPUs, one is non-GPUs.
            
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
                ERROR('Output directory exists, please change the name and restart the program', "mref_ali2d_MPI", 1, myid)

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
                    tmp_img.set_attr('ID',img_idx)
                    original_images[idx+i] = tmp_img.copy()
#                 original_images = EMData.read_images(args[0], list(range(batch_start, batch_end)))
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
                        send_EMData_w_ID( original_images[i], gpu, my_img, comm=mpi.MPI_COMM_WORLD )
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
                            original_images.append( recv_EMData_w_ID(proc, proc_img, comm=mpi.MPI_COMM_WORLD) )

            # step 2b: each non-GPU proc makes sure they sent off all their data
            else:
                assert len(original_images) == 0, "ERROR: proc[%d] still holds %d images." % (myid, len(original_images))
                image_start, image_end = None, None

            mpi.mpi_barrier(mpi.MPI_COMM_WORLD) # the above communication is blocking, but just to be sure

            #--------------------------------------------------[ run the GPU pre-alignment ]
            
            if Blockdata["myid_on_node"] in GPU_DEVICES:
            
                # 2D gpu alignment call
                if myid==0: print( time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "main() :: Executing pre-alignment" )

                params2d = mref_ali2d_gpu(
                    args[0],                          #filename
                    original_images,                  # downsampled images handled by this process only
                    args[1],                        #ref img stack
                    outdir,                         # output directory
                    mask,                             # mask
                    options.ir,                                # inner radius / first ring
                    options.ou,                    # outer radius / last ring
                    options.rs,                                # ring step
                    options.xr,                              # list of search rangees in x-dim
                    options.yr,                              # list of search rangees in y-dim
                    options.ts,                              # search step size
                    options.center,                    # centering method (should be 0)
                    options.maxit,                               # iteration limit
                    options.CTF,                      # CTF flag
                    options.snr,                              # snr (CTF parameter)
                    options.function,                      # user_func_name(?)
                    options.rand_seed,                               # random seed
                    mpi.mpi_comm_size(MPI_GPU_COMM),  # mpi comm size      |
                    mpi.mpi_comm_rank(MPI_GPU_COMM),  # mpi rank           |_______[ gpu communicator ]
                    0,                                # mpi main proc      |
                    MPI_GPU_COMM,                     # mpi communicator  _|
                    mpi_gpu_proc=(Blockdata["myid_on_node"] in GPU_DEVICES),
                    cuda_device_occ=0.9 )

                mpi.mpi_barrier(MPI_GPU_COMM)
                
                if myid==0: print(time.strftime("%Y-%m-%d %H:%M:%S :: ", time.localtime()) + "main() :: Pre-alignment call complete" )





    global_def.BATCH = False
    from mpi import mpi_finalize
    mpi_finalize()



if __name__ == "__main__":
    main()
