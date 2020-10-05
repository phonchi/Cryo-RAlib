#!/usr/local/EMAN2/bin/python
from __future__ import print_function

import os
import global_def
from global_def import *
from optparse import OptionParser
import sys

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
        #  even, odd, numer of even, number of images.  After frc, totav
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
        print_end_msg("mref_ali2d_MPI")

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



def main():
    arglist = []
    for arg in sys.argv:
        arglist.append( arg )
    progname = os.path.basename(sys.argv[0])
    usage = progname + " data_stack reference_stack outdir <maskfile> --ir=inner_radius --ou=outer_radius --rs=ring_step --xr=x_range --yr=y_range  --ts=translation_step --center=center_type --maxit=max_iteration --CTF --snr=SNR --function=user_function_name --rand_seed=random_seed --MPI"
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
    parser.add_option("--MPI", action="store_true", default=False,     help="  whether to use MPI version ")
    parser.add_option("--EQ", action="store_true", default=False,     help="  equal version ")
    (options, args) = parser.parse_args(arglist[1:])
    if len(args) < 3 or len(args) > 4:
            print("usage: " + usage)
            print("Please run '" + progname + " -h' for detailed options")
    else:
    
        if len(args) == 3:
            mask = None
        else:
            mask = args[3]

        if global_def.CACHE_DISABLE:
            from utilities import disable_bdb_cache
            disable_bdb_cache()
        
        if options.MPI:
            from mpi import mpi_init
            print("use mpi")
            sys.argv = mpi_init(len(sys.argv), sys.argv)

        global_def.BATCH = True
        if options.EQ:
            from development import mrefeq_ali2df
            #print  "  calling MPI",options.MPI,options.function,options.rand_seed
            #print  args
            mrefeq_ali2df(args[0], args[1], mask, options.ir, options.ou, options.rs, options.xr, options.yr, options.ts, options.center, options.maxit, options.CTF, options.snr, options.function, options.rand_seed, options.MPI)
        else:
            mref_ali2d(args[0], args[1], args[2], mask, options.ir, options.ou, options.rs, options.xr, options.yr, options.ts, options.center, options.maxit, options.CTF, options.snr, options.function, options.rand_seed, options.MPI)
        global_def.BATCH = False
        if options.MPI:
            from mpi import mpi_finalize
            mpi_finalize()


if __name__ == "__main__":
    main()
