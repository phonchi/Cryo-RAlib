
/******************************************************************************

GPU based reference free alignment

Author (C) 2019, Fabian Schoenfeld (fabian.schoenfeld@mpi-dortmund.mpg.de)
Copyright (C) 2019, Max Planck Institute of Molecular Physiology, Dortmund

   This program is free software: you can redistribute it and/or modify it 
under the terms of the GNU General Public License as published by the Free 
Software Foundation, either version 3 of the License, or (at your option) any
later version.

   This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along with
this program.  If not, please visit: http://www.gnu.org/licenses/

******************************************************************************/


//===================================================================[ header ]

#include "gpu_aln_noref.h"

//------------------------------------------------------------------[ globals ]

int CUDA_DEVICE_ID = -1;

struct AlignmentResources{

    const AlignConfig* aln_cfg = NULL;       // global alignment parameters

    // re-usable alignment resources
    float* u_polar_sample_coords   = NULL;   // re-usable polar sampling template
    vector<array<float,2>>* shifts = NULL;   // re-usable list of shifts to be applied per iteration
    AlignParam*   u_aln_param      = NULL;   // accumulated alignment parameters
    unsigned int* u_cid_idx        = NULL;   // class id (cid) index (idx) information
    
    // data handlers
    BatchHandler* ref_batch = NULL;          // reference handler ("reference" images are updated each iteration)
    BatchHandler* sbj_batch = NULL;          // subject handler ("subject" images are aligned to a given references)

} aln_res;

/*/////////////////////////////////////////////////////////////////////////////
// NOTE: Below is a pure debugging function that introduces additional
//       dependencies (opencv) and therefore is commented out by default.

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
extern "C" void extract_gpu_averages( const unsigned int idx ){
    unsigned int img_dim = aln_res.aln_cfg->img_dim;
    printf( "Writing %u averages from GPU texture memory..\n", aln_res.aln_cfg->ref_num );
    fflush(stdout); fflush(stderr);

    size_t tex_pitch;
    float h_tex_data[ img_dim*img_dim ];

    for( unsigned int i=0; i<aln_res.aln_cfg->ref_num; i++ ){
        float* d_tex_data = aln_res.ref_batch->get_tex_data( &tex_pitch );

        CUDA_ERR_CHK( cudaMemcpy2D(h_tex_data, img_dim*sizeof(float),
                      &d_tex_data[(tex_pitch/sizeof(float))*img_dim*i], tex_pitch,
                      img_dim*sizeof(float), img_dim,
                      cudaMemcpyDeviceToHost) );

        cv::Mat cv_img_mat( img_dim, img_dim, CV_32FC1, h_tex_data );
        double min, max;
        cv::minMaxLoc( cv_img_mat, &min, &max );
        cv_img_mat -= min;
        cv_img_mat /= max-min;
        cv_img_mat *= 255;
        cv::imwrite( cv::format("[%u]_gpu_avg_%u.png", i, idx), cv_img_mat );
    }
}
/////////////////////////////////////////////////////////////////////////////*/

//========================================================[ gpu alignment API ]

void gpu_alignment_init( const AlignConfig* aln_cfg, const unsigned int cuda_device_id ){
    /*
    Allocate the resources for multiple reference free alignment calls that
    share the same parameters (but not the same data). This is encapsulated
    here in order to avoid having to do it in each alignment call.

    Args:
        aln_cfg (const AlignConfig*): Structure holding all relevant parameters
            for any upcoming ref_free_alignment_2D() calls.

        cuda_device_id (const unsigned int): Specifies which GPU on the host
            machine is used.
    */
    
    // printf( "  -- CUDA LAYER --  :: gpu_alignment_init() :: Setting CUDA device %d\n", cuda_device_id ); // debug output; remove once "device ordinal" bug is fixed

    // first things first: set GPU
    assert( CUDA_DEVICE_ID == -1 || CUDA_DEVICE_ID == cuda_device_id );
    CUDA_DEVICE_ID = cuda_device_id;
    CUDA_ERR_CHK( cudaSetDevice(cuda_device_id) );

    // set global alignment parameters
    aln_res.aln_cfg = aln_cfg;

    // create re-usable polar sampling template
    aln_res.u_polar_sample_coords = generate_polar_sampling_points( aln_cfg->ring_num, aln_cfg->ring_len );

    // create re-usable list of shifts to be applied per iteration
    aln_res.shifts = generate_shift_array( aln_cfg->shift_rng_x, aln_cfg->shift_rng_y, aln_cfg->shift_step );
}

extern "C" void reset_shifts( const float shift_range, const float shift_step ){
    /* Constructs a new shift array. Note that we have to make sure that the
    new array has the same length as the existing one, since otherwise all the
    offsets in our multiplication table are fucked. Can't have that.

    Notes to self on shift use:
    - Device only ever gets passed shift values, never the shift array.
    - BatchHandler never uses any shift vals, only passes them along.
    - CcfResultTable only uses shift_range & shift_step during construction to
      compute shift_num. This is the only value it keeps and cares about (used
      to compute ccf table offsets during aln_param extraction).
    */

    assert( aln_res.shifts != NULL ); // sanity check: shift reset should only ever be done after *_init()

    vector<array<float,2>>* new_shifts = generate_shift_array( shift_range, shift_range, shift_step );
    assert( aln_res.shifts->size() == new_shifts->size() );

    delete( aln_res.shifts );
    aln_res.shifts = new_shifts;
}

extern "C" void gpu_clear(){

    if( aln_res.u_polar_sample_coords != NULL ){
        CUDA_ERR_CHK( cudaFree(aln_res.u_polar_sample_coords) );
        aln_res.u_polar_sample_coords = NULL;
    }

    if( aln_res.u_aln_param != NULL ){
        CUDA_ERR_CHK( cudaFree(aln_res.u_aln_param) );
        aln_res.u_aln_param = NULL;
    }

    if( aln_res.u_cid_idx != NULL ){
        CUDA_ERR_CHK( cudaFree(aln_res.u_cid_idx) );
        aln_res.u_cid_idx = NULL;
    }

    if( aln_res.shifts != NULL ){
        delete( aln_res.shifts );
        aln_res.shifts = NULL;
    }

    if( aln_res.ref_batch ){
        delete( aln_res.ref_batch );
        aln_res.ref_batch = NULL;
    }
    if( aln_res.sbj_batch ){
        delete( aln_res.sbj_batch );
        aln_res.sbj_batch = NULL;
    }

    /* 
    Optional: CUDA_ERR_CHK( cudaDeviceReset() );
    This will destroy ALL allocated resources on the device INCLUDING those
    allocated by other processes. It's fun but in one test ran way too slow.
    */
}

//------------------------------------------------------------[ pre-alignment ]

/* PRE-ALIGNMENT
 - Large image stack that will not fit in GPU memory.
 - All images are aligned to a single reference (ie belong to a single class).
 - Reference is updated on the Python end once every sbj img has been aligned.
 - This repeats a given number of times.
 */

extern "C" AlignParam* pre_align_init(
    const unsigned int num_particles,
    const AlignConfig* aln_cfg,
    const unsigned int cuda_device_id )
{
    /*
    Allocate the resources for multiple reference free alignment calls that
    share the same parameters (but not the same data). This is encapsulated
    here in order to avoid having to do it in each alignment call.

    NOTE: In order to be able to make use of the returned pointer in Python
    we need to specify its type in Python using:
       cu_module.pre_align_init.restype = ctypes.c_ulonglong

    Args:
        aln_cfg (const AlignConfig*): Structure holding all relevant parameters
            for any upcoming ref_free_alignment_2D() calls.

        ref_data_list (const float**): Array containint reference image data.

        cuda_device_id (const unsigned int): Specifies which GPU on the host
            machine is used.
    */

    // generic gpu alignment init
    gpu_alignment_init( aln_cfg, cuda_device_id );

    // particle class id (during pre-alignment we only have a single class)
    CUDA_ERR_CHK( cudaMallocManaged(&(aln_res.u_aln_param), num_particles*sizeof(AlignParam)) );
    for( unsigned int i=0; i<num_particles; i++ )
        aln_res.u_aln_param[i].ref_id = 0;

    // Create class id (cid) index (idx) list in unified memory. For pre-
    // alignment this is simply [0,limit] since we only have one reference
    CUDA_ERR_CHK( cudaMallocManaged(&aln_res.u_cid_idx, (aln_cfg->ref_num+1)*sizeof(unsigned int)) );
    aln_res.u_cid_idx[0] = 0;
    aln_res.u_cid_idx[1] = aln_cfg->sbj_num;

    // initialize batch handlers
    aln_res.sbj_batch = new BatchHandler( aln_cfg, "subject_batch" );
    aln_res.ref_batch = new BatchHandler( aln_cfg, "pre_align_ref_batch" );

    // all done
    return aln_res.u_aln_param;
}

extern "C" bool pre_align_size_check(
    const unsigned int num_particles,
    const AlignConfig* cfg,
    const unsigned int cuda_device_id,
    const float        request,
    const bool         verbose )
{
    // first things first: set GPU
    CUDA_ERR_CHK( cudaSetDevice(cuda_device_id) );

    // sanity checks
    assert( sizeof(unsigned int) >= 4 );
    assert( sizeof(size_t) >= 8 );

    if(verbose) printf( "GPU[%d] SIZE CHECK\n", cuda_device_id );

    //--------------------------------------------------------------[ globals ]

    size_t polar_template = (cfg->ring_len*cfg->ring_num*2) * sizeof(float);
    size_t shift_array = (cfg->shift_rng_x+cfg->shift_rng_y+1)*2 * sizeof( unsigned int );
    size_t aln_param = sizeof(AlignParam) * num_particles;  // *
    size_t cid_idx = (cfg->ref_num+1) * sizeof(unsigned int);

    size_t globals = (polar_template + shift_array + aln_param + cid_idx) / (1024*1024);
    if(verbose) printf( "  in/out param:   %zuMB\n", globals );

    // * NOTE: In pre-alignment the length of the aln_param list is constant (num_particles,
    //  to be precise) and we only vary the batch size of particles processed on the GPU.

    //------------------------------------------------[ subject batch handler ]

    size_t img_mem_pitch = get_device_mem_pitch(cuda_device_id);
    img_mem_pitch = ((cfg->img_dim*sizeof(float)) / img_mem_pitch + 1) * img_mem_pitch;
    size_t tex_memory = (img_mem_pitch*cfg->img_dim*cfg->sbj_num) / (1024*1024);
    if(verbose) printf( "  sbj tex memory: %zuMB\n", tex_memory );

    size_t polar_memory = (cfg->sbj_num * (cfg->ring_len+2) * cfg->ring_num * sizeof(float)) / (1024*1024);
    size_t fft_memory   = (cfg->sbj_num * (cfg->img_dim+2)  * cfg->img_dim  * sizeof(float)) / (1024*1024);
    size_t sbj_data_mem = max( polar_memory, fft_memory );
    if(verbose) printf( "  sbj dat memory: %zuMB\n", sbj_data_mem );

    assert( fmod(cfg->shift_rng_x, cfg->shift_step) == 0.0f );
    assert( fmod(cfg->shift_rng_y, cfg->shift_step) == 0.0f );
    size_t shift_num = (2*(cfg->shift_rng_x/cfg->shift_step)+1) * (2*(cfg->shift_rng_y/cfg->shift_step)+1);

    size_t stack_table_size = ((cfg->ring_len+2) * cfg->sbj_num * shift_num*2 * sizeof(float)) / (1024*1024);
    stack_table_size *= 1; // only one global avg used as reference in the pre-alignment
    if(verbose) printf( "  ccf table mem:  %zuMB\n", stack_table_size );

    size_t data_arrays = (cfg->sbj_num*sizeof(int) + cfg->sbj_num*2*sizeof(float)) / (1024*1024);
    if(verbose) printf( "  sbj transfer:   %zuMB\n", data_arrays );


    ///////////////////////////////////
    // check for early abort (estimated size is too large already)
    size_t cufft_estimate_ccf, cufft_estimate_sbj;
    CUFFT_ERR_CHK( cufftEstimate1d(cfg->ring_len, CUFFT_C2R, cfg->sbj_num*1*shift_num*2, &cufft_estimate_ccf) );
    CUFFT_ERR_CHK( cufftEstimate1d(cfg->ring_len, CUFFT_R2C, cfg->ring_num*cfg->sbj_num, &cufft_estimate_sbj) );
    cufft_estimate_ccf /= 1024*1024;
    cufft_estimate_sbj /= 1024*1024;

    size_t mem_avl=0, mem_ttl=0;
    CUDA_ERR_CHK( cudaMemGetInfo(&mem_avl, &mem_ttl) );
    mem_avl /= 1024*1024;
    
    size_t sub_total = globals + tex_memory + sbj_data_mem + stack_table_size + cufft_estimate_ccf + cufft_estimate_sbj + data_arrays;
    if(sub_total >= mem_avl*request) return false;
    ///////////////////////////////////


    cufftHandle cufft_pln;
    size_t cufft_workspace_size_ccf;
    CUFFT_ERR_CHK( cufftPlan1d(&cufft_pln, cfg->ring_len, CUFFT_C2R, cfg->sbj_num*1*shift_num*2) ); // cfg->ref_num==1 for ref free alignment
    CUFFT_ERR_CHK( cufftGetSize(cufft_pln, &cufft_workspace_size_ccf) );
    CUFFT_ERR_CHK( cufftDestroy(cufft_pln) );
    cufft_workspace_size_ccf /= 1024*1024;
    if(verbose) printf( "  ccf cufft size: %zuMB\n", cufft_workspace_size_ccf );

    size_t cufft_workspace_size_sbj;
    CUFFT_ERR_CHK( cufftPlan1d(&cufft_pln, cfg->ring_len, CUFFT_R2C, cfg->ring_num*cfg->sbj_num) );
    CUFFT_ERR_CHK( cufftGetSize(cufft_pln, &cufft_workspace_size_sbj) );
    CUFFT_ERR_CHK( cufftDestroy(cufft_pln) );
    cufft_workspace_size_sbj /= 1024*1024;
    if(verbose) printf( "  sbj cufft size: %zuMB\n", cufft_workspace_size_sbj );

    size_t sbj_handler = tex_memory + sbj_data_mem + stack_table_size + cufft_workspace_size_ccf + cufft_workspace_size_sbj + data_arrays;
    if(verbose) printf( "SBJ BATCH HANDLER: %zuMB\n", sbj_handler );

    //----------------------------------------------[ reference batch handler ]

    tex_memory = (img_mem_pitch * cfg->img_dim * cfg->ref_num) / (1024*1024);
    if(verbose) printf( "  ref tex memory: %zuMB\n", tex_memory );

    polar_memory = (cfg->ref_num * (cfg->ring_len+2) * sizeof(float) * cfg->ring_num) / (1024*1024);
    if(verbose) printf( "  ref plr memory: %zuMB\n", polar_memory );

    size_t ref_handler = tex_memory + polar_memory;
    if(verbose) printf( "REF BATCH HANDLER: %zuMB\n", ref_handler );

    //------------------------------------------------------------[ mem check ]

    size_t total = globals + sbj_handler + ref_handler;
    if(verbose) printf( "TOTAL: %zuMB\n", total );

    // mem check
    CUDA_ERR_CHK( cudaMemGetInfo(&mem_avl, &mem_ttl) );
    mem_avl /= 1024*1024;

    // ship it
    if(mem_avl*request >= total)
        return true;
    else
        return false;
}

extern "C" void pre_align_fetch(
    const float**      img_data,
    const unsigned int img_num,
    const char*        batch_select )
{
    // select batch to fill    
    BatchHandler* batch_handler = NULL;
    if( strcmp(batch_select, "sbj_batch")==0 ) batch_handler = aln_res.sbj_batch;
    if( strcmp(batch_select, "ref_batch")==0 ) batch_handler = aln_res.ref_batch;
    
    if( batch_handler == NULL ){
        printf( "ERROR! fetch_data() :: Unknown batch type \'%s\' specified.\n", batch_select );
        return;
    }

    // transfer data
    batch_handler->fetch_data( img_data, 0, img_num );
}


extern "C" void mref_align_run( const int start_idx, const int stop_idx, const int cnx, const int cny ){

    //----------------------------------------------------------------[ setup ]

    // grab batch handlers
    BatchHandler* ref_batch = aln_res.ref_batch;
    BatchHandler* sbj_batch = aln_res.sbj_batch;

    // update reference
    ref_batch->resample_to_polar( 0, 0, 0, aln_res.u_polar_sample_coords );
    ref_batch->apply_FFT();

    //------------------------------------------------------------[ alignment ]

    for( unsigned int shift_idx=0; shift_idx < aln_res.shifts->size(); shift_idx++ ){
        sbj_batch->resample_to_polar( 
            (*aln_res.shifts)[shift_idx][0]+cnx,
            (*aln_res.shifts)[shift_idx][1]+cny, start_idx,
            aln_res.u_polar_sample_coords );
        sbj_batch->apply_FFT();
        sbj_batch->ccf_mult_m( ref_batch, shift_idx, 0 );
    }
    sbj_batch->apply_IFFT();
    CUDA_ERR_CHK( cudaDeviceSynchronize() );

    sbj_batch->compute_alignment_param( start_idx, stop_idx, aln_res.shifts, aln_res.u_aln_param );
}


extern "C" void pre_align_run( const int start_idx, const int stop_idx ){

    //----------------------------------------------------------------[ setup ]

    // grab batch handlers
    BatchHandler* ref_batch = aln_res.ref_batch;
    BatchHandler* sbj_batch = aln_res.sbj_batch;

    // update reference
    ref_batch->resample_to_polar( 0, 0, 0, aln_res.u_polar_sample_coords );
    ref_batch->apply_FFT();

    //------------------------------------------------------------[ alignment ]

    for( unsigned int shift_idx=0; shift_idx < aln_res.shifts->size(); shift_idx++ ){
        sbj_batch->resample_to_polar( 
            (*aln_res.shifts)[shift_idx][0],
            (*aln_res.shifts)[shift_idx][1], start_idx,
            aln_res.u_polar_sample_coords );
        sbj_batch->apply_FFT();
        sbj_batch->ccf_mult( ref_batch, shift_idx, 0 );
    }
    sbj_batch->apply_IFFT();
    CUDA_ERR_CHK( cudaDeviceSynchronize() );

    sbj_batch->compute_alignment_param( start_idx, stop_idx, aln_res.shifts, aln_res.u_aln_param );
}

//-------------------------------------------------[ reference-free alignment ]

/* REFERENCE-FREE ALIGNMENT
 - Large image stack where continuous section belong to the same class.
 - All images of a class are aligned to the same reference.
 - Class assignment does not change over the course of alignment iterations.
 - References are updated directly on the device.
 - Once as many class-averages as we can fit in gpu memory have been produced,
   we swap out the data with another set of classes and repeat until done.
 */

extern "C" AlignParam* ref_free_alignment_2D_init( 
    const AlignConfig* aln_cfg,
    const float**      sbj_data_list,
    const float**      ref_data_list,
    const int*         sbj_cid_list,
    const unsigned int cuda_device_id )
{
    /*
    Allocate the resources for multiple reference free alignment calls that
    share the same parameters (but not the same data). This is encapsulated
    here in order to avoid having to do it in each alignment call.

    NOTE: In order to be able to make use of the returned pointer in Python
    we need to specify its type in Python using:
       cu_module.ref_free_alignment_2D_init.restype = ctypes.c_ulonglong

    Args:
        aln_cfg (const AlignConfig*): Structure holding all relevant parameters
            for any upcoming ref_free_alignment_2D() calls.

        sbj_data_list (const float**): Array containing the subject image data.
            (Subject images are the particle images that we align to their
            references.) Image dimensions are listed in the <aln_cfg> struct.

        ref_data_list (const float**): Array containint reference image data.

        sbj_cid_list (const int*): Contains as many entries as we have subject
            images and assigns a reference/class to each subject (e.g. the
            the relevant reference for the image stored in sbj_data_list[n]
            has index sbj_cid_list[n] and is accessed via red_data_list[n]).

        cuda_device_id (const unsigned int): Specifies which GPU on the host
            machine is used.
    */

    // generic gpu alignment init
    gpu_alignment_init( aln_cfg, cuda_device_id );

    // alignment parameters (unified memory)
    CUDA_ERR_CHK( cudaMallocManaged(&(aln_res.u_aln_param), aln_cfg->sbj_num*sizeof(AlignParam)) );
    for( unsigned int i=0; i<aln_cfg->sbj_num; i++ )
        aln_res.u_aln_param[i].ref_id = sbj_cid_list[i];

    // create class id (cid) index (idx) list in unified memory
    CUDA_ERR_CHK( cudaMallocManaged(&aln_res.u_cid_idx, (aln_cfg->ref_num+1)*sizeof(unsigned int)) );
    aln_res.u_cid_idx[aln_cfg->ref_num] = aln_cfg->sbj_num;
    int cid=-1, idx=0;
    for( unsigned int i=0; i<aln_cfg->sbj_num; i++ ){
        if( sbj_cid_list[i] != cid ){
            aln_res.u_cid_idx[idx] = i;
            cid = sbj_cid_list[i];
            idx += 1;
        }
    }

    // initialize batch handlers
    aln_res.sbj_batch = new BatchHandler( aln_cfg, "subject_batch"   );
    aln_res.ref_batch = new BatchHandler( aln_cfg, "reference_batch" );

    aln_res.sbj_batch->fetch_data( sbj_data_list, 0, aln_res.aln_cfg->sbj_num );
    aln_res.ref_batch->fetch_data( ref_data_list, 0, aln_res.aln_cfg->ref_num );

    // all done
    return aln_res.u_aln_param;
}

extern "C" bool ref_free_alignment_2D_size_check( 
    const AlignConfig* cfg,
    const unsigned int cuda_device_id,
    const float        request,
    const bool         verbose )
{
    // first things first: set GPU
    CUDA_ERR_CHK( cudaSetDevice(cuda_device_id) );

    // sanity checks
    assert( sizeof(unsigned int) >= 4 );
    assert( sizeof(size_t) >= 8 );

    if(verbose) printf( "GPU[%d] SIZE CHECK\n", cuda_device_id );

    //--------------------------------------------------------------[ globals ]

    size_t polar_template = (cfg->ring_len*cfg->ring_num*2) * sizeof(float);
    size_t shift_array = (cfg->shift_rng_x+cfg->shift_rng_y+1)*2 * sizeof( unsigned int );
    size_t aln_param = sizeof(AlignParam) * cfg->sbj_num;
    size_t cid_idx = (cfg->ref_num+1) * sizeof(unsigned int);

    size_t globals = (polar_template + shift_array + aln_param + cid_idx) / (1024*1024);
    if(verbose) printf( "  in/out param:   %zuMB\n", globals );

    //------------------------------------------------[ subject batch handler ]

    size_t img_mem_pitch = get_device_mem_pitch(cuda_device_id);
    img_mem_pitch = ((cfg->img_dim*sizeof(float)) / img_mem_pitch + 1) * img_mem_pitch;
    size_t tex_memory = (img_mem_pitch*cfg->img_dim*cfg->sbj_num) / (1024*1024);
    if(verbose) printf( "  sbj tex memory: %zuMB\n", tex_memory );

    size_t polar_memory = (cfg->sbj_num * (cfg->ring_len+2) * sizeof(float) * cfg->ring_num) / (1024*1024);
    size_t fft_memory   = (cfg->sbj_num * (cfg->img_dim+2) * cfg->img_dim * sizeof(float))   / (1024*1024);
    size_t sbj_data_mem = max( polar_memory, fft_memory );
    if(verbose) printf( "  sbj dat memory: %zuMB\n", sbj_data_mem );

    assert( fmod(cfg->shift_rng_x, cfg->shift_step) == 0 );
    assert( fmod(cfg->shift_rng_y, cfg->shift_step) == 0 );
    size_t shift_num = (2*(cfg->shift_rng_x/cfg->shift_step)+1) * (2*(cfg->shift_rng_y/cfg->shift_step)+1);
    
    size_t stack_table_size = ((cfg->ring_len+2) * cfg->sbj_num * shift_num*2*sizeof(float)) / (1024*1024);
    stack_table_size *= 1; // only one reference slot in the ccf table for ref free aligment
    if(verbose) printf( "  ccf table mem:  %zuMB\n", stack_table_size );

    size_t data_arrays = (cfg->sbj_num*sizeof(int) + cfg->sbj_num*2*sizeof(float)) / (1024*1024);
    if(verbose) printf( "  sbj transfer:   %zuMB\n", data_arrays );


    ///////////////////////////////////
    // check for early abort (estimated size is too large already)
    size_t cufft_estimate_ccf, cufft_estimate_sbj;
    CUFFT_ERR_CHK( cufftEstimate1d(cfg->ring_len, CUFFT_C2R, cfg->sbj_num*1*shift_num*2, &cufft_estimate_ccf) );
    CUFFT_ERR_CHK( cufftEstimate1d(cfg->ring_len, CUFFT_R2C, cfg->ring_num*cfg->sbj_num, &cufft_estimate_sbj) );
    cufft_estimate_ccf /= 1024*1024;
    cufft_estimate_sbj /= 1024*1024;

    size_t mem_avl=0, mem_ttl=0;
    CUDA_ERR_CHK( cudaMemGetInfo(&mem_avl, &mem_ttl) );
    mem_avl /= 1024*1024;

    size_t sub_total = globals + tex_memory + sbj_data_mem + stack_table_size + cufft_estimate_ccf + cufft_estimate_sbj + data_arrays;
    if(sub_total >= mem_avl*request) return false;
    ///////////////////////////////////


    cufftHandle cufft_pln;
    size_t cufft_workspace_size_ccf;
    CUFFT_ERR_CHK( cufftPlan1d(&cufft_pln, cfg->ring_len, CUFFT_C2R, cfg->sbj_num*1*shift_num*2) ); // cfg->ref_num==1 for ref free alignment
    CUFFT_ERR_CHK( cufftGetSize(cufft_pln, &cufft_workspace_size_ccf) );
    CUFFT_ERR_CHK( cufftDestroy(cufft_pln) );
    cufft_workspace_size_ccf /= 1024*1024;
    if(verbose) printf( "  ccf cufft size: %zuMB\n", cufft_workspace_size_ccf );

    size_t cufft_workspace_size_sbj;
    CUFFT_ERR_CHK( cufftPlan1d(&cufft_pln, cfg->ring_len, CUFFT_R2C, cfg->ring_num*cfg->sbj_num) );
    CUFFT_ERR_CHK( cufftGetSize(cufft_pln, &cufft_workspace_size_sbj) );
    CUFFT_ERR_CHK( cufftDestroy(cufft_pln) );
    cufft_workspace_size_sbj /= 1024*1024;
    if(verbose) printf( "  sbj cufft size: %zuMB\n", cufft_workspace_size_sbj );

    size_t sbj_handler = tex_memory + sbj_data_mem + stack_table_size + cufft_workspace_size_ccf + cufft_workspace_size_sbj + data_arrays;
    if(verbose) printf( "SBJ BATCH HANDLER: %zuMB\n", sbj_handler );

    //----------------------------------------------[ reference batch handler ]

    tex_memory = (img_mem_pitch * cfg->img_dim * cfg->ref_num) / (1024*1024);
    if(verbose) printf( "  ref tex memory: %zuMB\n", tex_memory );

    polar_memory = (cfg->ref_num * (cfg->ring_len+2) * sizeof(float) * cfg->ring_num) / (1024*1024);
    if(verbose) printf( "  ref plr memory: %zuMB\n", polar_memory );

    size_t cufft_workspace_size;
    CUFFT_ERR_CHK( cufftPlan1d(&cufft_pln, cfg->ring_len, CUFFT_R2C, cfg->ring_num*cfg->ref_num) );
    CUFFT_ERR_CHK( cufftGetSize(cufft_pln, &cufft_workspace_size) );
    CUFFT_ERR_CHK( cufftDestroy(cufft_pln) );
    cufft_workspace_size /= 1024*1024;
    if(verbose) printf( "  cufft req size: %zuMB\n", cufft_workspace_size );

    size_t ref_handler = tex_memory + polar_memory + cufft_workspace_size;
    if(verbose) printf( "REF BATCH HANDLER: %zuMB\n", ref_handler );

    //------------------------------------------------------------[ mem check ]

    unsigned int total = globals + sbj_handler + ref_handler;
    if(verbose) printf( "TOTAL: %uMB\n", total );

    // mem check
    CUDA_ERR_CHK( cudaMemGetInfo(&mem_avl, &mem_ttl) );
    mem_avl /= 1024*1024;

    // ship it
    if(mem_avl*request >= total)
        return true;
    else
        return false;
}

extern "C" void ref_free_alignment_2D(){

    //----------------------------------------------------------------[ setup ]

    // grab alignment parameters
    const AlignConfig* cfg = aln_res.aln_cfg;

    // grab batch handlers
    BatchHandler* ref_batch = aln_res.ref_batch;
    BatchHandler* sbj_batch = aln_res.sbj_batch;

    // update references
    ref_batch->resample_to_polar( 0, 0, 0, aln_res.u_polar_sample_coords );
    ref_batch->apply_FFT();

    //------------------------------------------------------------[ alignment ]

    for( unsigned int shift_idx=0; shift_idx < aln_res.shifts->size(); shift_idx++ ){
        sbj_batch->resample_to_polar( 
            (*aln_res.shifts)[shift_idx][0],
            (*aln_res.shifts)[shift_idx][1], 0,
            aln_res.u_polar_sample_coords );
        sbj_batch->apply_FFT();
        sbj_batch->ccf_mult( ref_batch, shift_idx, 0 );
    }
    sbj_batch->apply_IFFT();

    CUDA_ERR_CHK( cudaDeviceSynchronize() );

    sbj_batch->compute_alignment_param( 0, cfg->sbj_num, aln_res.shifts, aln_res.u_aln_param );
    sbj_batch->apply_alignment_param( aln_res.u_aln_param );  // NOTE: includes device sync at the end
    ref_batch->fetch_averages( sbj_batch->img_ptr(0) );       // NOTE: includes device sync at the end
}

extern "C" void ref_free_alignment_2D_filter_references( 
    const float cutoff_freq,
    const float falloff )
{
    aln_res.ref_batch->apply_tangent_filter( cutoff_freq, falloff );
}

//=============================================================[ cuda kernels ]

__global__ void cu_apply_tanl_filter_to_tex(
    cuComplex*         tgt_tex_data,        // take the images stored in this texture
    const size_t       tgt_tex_pitch,       // note that we don't need image width but texture pitch
    const unsigned int img_dim_y,           // though for number of rows the y-dim is just fine
    const float        filter_cutoff_freq,  // apply tanh filter with this cutoff frequency
    const float        filter_fallof )      // and this falloff
{

    unsigned int img_idx     = blockIdx.x;   // [0,num_img-1]
    unsigned int img_coord_x = threadIdx.x;  // [0,img_dim_x/2+1] (note we are in freq. space)

    tgt_tex_data = &tgt_tex_data[ img_idx * img_dim_y*(tgt_tex_pitch/sizeof(cuComplex)) ];

    float c = PI / (2*filter_fallof*filter_cutoff_freq);    // tangent filter constant term
    float x = float(img_coord_x)/float(blockDim.x) * 0.5f;  // scale x-coords down to [0.0, 0.5]
    float img_dim_y_half = float(img_dim_y)/2.0f;

    for( unsigned int img_coord_y=0; img_coord_y<img_dim_y; img_coord_y++ ){

        float y = (img_coord_y < img_dim_y_half) ? float(img_coord_y) : float(img_dim_y-img_coord_y);
        y = y/img_dim_y_half*0.5f;

        //float y = fabsf(img_coord_y - img_dim_y_half)/img_dim_y_half * 0.5f;  // scale y-coords down to [0.0, 0.5]
        float d = sqrtf(x*x + y*y);

        unsigned int write_idx = img_coord_y*(tgt_tex_pitch/sizeof(cuComplex)) + img_coord_x;
        
        tgt_tex_data[ write_idx ].x *= 0.5f * ( tanhf(c*(d+filter_cutoff_freq)) - tanhf(c*(d-filter_cutoff_freq)) );
        tgt_tex_data[ write_idx ].y *= 0.5f * ( tanhf(c*(d+filter_cutoff_freq)) - tanhf(c*(d-filter_cutoff_freq)) );
    }
}

__global__ void cu_resample_to_polar(
    const cudaTextureObject_t img_tex,         // take the images stored in this texture
    const AlignParam*  u_aln_param,            // apply the individual image shifts stored here
    float*             d_polar_data,           // and put the re-encoded polar images here
    const unsigned int img_dim_x,              // images have this x-dimension
    const unsigned int img_dim_y,              // and these y-dimensions
    const int          shift_x,                // in addition to the individual shift, also
    const int          shift_y,                // apply these global shift values
    const float*       u_polar_sample_coords,  // use these pre-computed sampling coordinates for the polar encoding
    const unsigned int ring_len,               // use this ring length for the polar encoding
    const unsigned int ring_num )              // use this number of rings for the polar encoding
{
    /* SUMMARY
    <img_num> x <ring_num> blocks in total where <ring_num> blocks process one
    image. Each block contains <ring_len> threads where each of which computes
    one value of the polar representation of an image.
    */

    /* NOTE
    The width of the <u_polar_sample_coords> buffer is <ring_len+2>. The polar
    representation only fills a widh of <ring_len> though, as this is enough
    for storing it. The additional size of +2 per row will be needed later when
    we do an in-place FFT of the polar representation.
       This is why we compute <ring_len> elements using <ring_len> threads but
    skip (ring_len+2) values when indexing the individual rows of the buffer.
    */

    /* NOTE
    Since we're using linear interpolation when accessing the data stored in
    textures we need an additional offset of +0.5 when indexing the data. This
    is because the color information of pixel[i] is found at the center of the
    pixel and needs to be indexed as pixel[i+0.5].
    */

    // grid ids
    unsigned int bid_x = blockIdx.x;   // determines the image to process [0,..,img_num-1]
    unsigned int bid_y = blockIdx.y;   // determines the ring to fill [0,..,ring_num-1]
    unsigned int tid   = threadIdx.x;  // determines the position within the ring to fill [0,..,ring_len-1]

    // shift images
    float cnt_x, cnt_y;
    if( u_aln_param != NULL ){
        cnt_x = img_dim_x/2 + shift_x + u_aln_param[bid_x].shift_x + 0.5;
        cnt_y = img_dim_y/2 + shift_y + u_aln_param[bid_x].shift_y + 0.5;
    }
    else{
        cnt_x = img_dim_x/2 + shift_x + 0.5;
        cnt_y = img_dim_y/2 + shift_y + 0.5;
    }

    // polar sampling points (offsets from image center)
    unsigned int polar_idx = 2* (bid_y*ring_len + tid);
    float polar_offset_x = u_polar_sample_coords[ polar_idx+0 ];
    float polar_offset_y = u_polar_sample_coords[ polar_idx+1 ];

    // each thread computes one value of the polar representation of an image
    unsigned int polar_data_idx = bid_x*ring_num*(ring_len+2) +    // select image slot
                                  bid_y*(ring_len+2) +             // select ring slot
                                  tid;                             // select element in ring

    d_polar_data[ polar_data_idx ] = tex2D<float>( img_tex, cnt_x+polar_offset_x, bid_x*img_dim_y + cnt_y+polar_offset_y );
}

__global__ void cu_ccf_mult( 
    const float*       sbj_batch_ptr, 
    const float*       ref_batch_ptr, 
    const AlignParam*  u_aln_param,
    float*             batch_table_row_ptr, 
    const unsigned int batch_table_row_offset, 
    const unsigned int batch_table_mirror_offset,
    const unsigned int ring_num )
{
    /*
    This kernel is used to compute the cross correlation function of two images, which can be expressed
    as FFT(img_a)' x FFT(img_b). This computation boils down to a number of scalar products with an
    additional factor for weighting the individual rings of the polar representation of the data.
    Note that the function parameters are float* but the values they are pointing to are interpreted as
    complex values -- i.e. every two floats denote a single complex value's real and imaginary parts.
    For details on the cross correlation, see: http://paulbourke.net/miscellaneous/correlate/
    For a walkthrough to the computation, see: https://pastebin.com/zVi13qRZ
    */

    /* Pattern of memory writes

    Each block processes one sbj img and computes its ccf values with the given ref img (the latter is the
    same across all blocks). Note that, at this point, the data is in complex values so the below code
    implements a complex value multiplication.
       We also immediately compute the values for the mirrored subject image. This can be done by multi-
    plying using the complex conjugate (representing a y-axis flipped image). Consequently, each thread
    computes two multiplications and writes two complex result values (four float values).

    Memory layout and access pattern during kernel execution is as follows:

      __ <batch_table_row_ptr> address indicates reference slot (+--+) to be filled in row_0
     /
    V
    [--+--+--+--] .. [--+--+--+--] || [--+--+--+--] .. [--+--+--+--]    First kernel call, given shift_0:
    [--+--+--+--] .. [--+--+--+--] || [--+--+--+--] .. [--+--+--+--]    compute ccf of all sbj imgs and ref_0
    :           :    :           :    :           :                :
    [--+--+--+--] .. [--+--+--+--] || [--+--+--+--] .. [--+--+--+--]

       V
    [==+--+--+--] .. [--+--+--+--] || [==+--+--+--] .. [--+--+--+--]    Second kernel call, given shift_0:
    [==+--+--+--] .. [--+--+--+--] || [==+--+--+--] .. [--+--+--+--]    compute ccf of all sbj imgs and ref_1
    :           :    :           :    :           :                :
    [==+--+--+--] .. [--+--+--+--] || [==+--+--+--] .. [--+--+--+--]

          V
    [==+==+--+--] .. [--+--+--+--] || [==+==+--+--] .. [--+--+--+--]    Third kernel call, given shift_0:
    [==+==+--+--] .. [--+--+--+--] || [==+==+--+--] .. [--+--+--+--]    compute ccf of all sbj imgs and ref_3
    :           :    :           :    :           :                :
    [==+==+--+--] .. [--+--+--+--] || [==+==+--+--] .. [--+--+--+--]

    (..)

    [==+==+==+==] .. [--+--+--+--] || [==+==+==+==] .. [--+--+--+--]    After final kernel call, given shift_0:
    [==+==+==+==] .. [--+--+--+--] || [==+==+==+==] .. [--+--+--+--]    compute ccf of all sbj imgs and ref_N
    :           :    :           :    :           :    :           :
    [==+==+==+==] .. [--+--+--+--] || [==+==+==+==] .. [--+--+--+--]

    (..)

    [==+==+==+==] .. [==+==+==+==] || [==+==+==+==] .. [==+==+==+==]    After final kernel call, given all
    [==+==+==+==] .. [==+==+==+==] || [==+==+==+==] .. [==+==+==+==]    shifts.
    :           :    :           :    :           :                :
    [==+==+==+==] .. [==+==+==+==] || [==+==+==+==] .. [==+==+==+==]

    \__/ blockDim.x*2                              For each given shift, kernel will be called once per
                                                   reference. Afterwards, the batch_table column block for 
    \___________/ blockDim.x*2 * ref_num           the current x/y shift is filled w/ the ccf results of 
                                                   each subject image for all references.
    */

    // grid indices
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;

    // each block picks their reference (NOTE: blockDim.x*2 == ring_len+2)
    ref_batch_ptr = &ref_batch_ptr[ u_aln_param[bid].ref_id * (blockDim.x*2)*ring_num ];

    // each block picks their image (NOTE: blockDim.x*2 == ring_len+2)
    unsigned int sbj_idx = bid * blockDim.x*2*ring_num;  // img_index x img_size

    float ccf_orig_r=0.0f, ccf_orig_i=0.0f, ccf_mirr_r=0.0f, ccf_mirr_i=0.0f;
    for( unsigned int i=0; i<ring_num; i++ ){

        // determine element offsets
        unsigned int ref_elem_idx = i*blockDim.x*2 + tid*2;  // select element in reference image
        unsigned int sbj_elem_idx = sbj_idx + ref_elem_idx;  // select element in subject image
        // read out element data
        float rr = ref_batch_ptr[ ref_elem_idx+0 ];  // ref elem real part
        float ri = ref_batch_ptr[ ref_elem_idx+1 ];  // ref elem imag part
        float sr = sbj_batch_ptr[ sbj_elem_idx+0 ];  // sbj elem real part
        float si = sbj_batch_ptr[ sbj_elem_idx+1 ];  // sbj elem imag part
        // tmps
        float rr_sr = rr * sr;
        float ri_si = ri * si;
        float rr_si = rr * si;
        float ri_sr = ri * sr;
        // weighted ccf values (for original and mirrored sbj image)
        ccf_orig_r += ( rr_sr+ri_si) * (i+1);
        ccf_orig_i += (-rr_si+ri_sr) * (i+1);
        ccf_mirr_r += ( rr_sr-ri_si) * (i+1);
        ccf_mirr_i += (-rr_si-ri_sr) * (i+1);
    }

    // write results
    unsigned int table_idx = blockIdx.x*batch_table_row_offset + tid*2;  // select row and elem_offset
    batch_table_row_ptr[ table_idx+0 ] = ccf_orig_r;
    batch_table_row_ptr[ table_idx+1 ] = ccf_orig_i;
    batch_table_row_ptr[ table_idx+batch_table_mirror_offset+0 ] = ccf_mirr_r;
    batch_table_row_ptr[ table_idx+batch_table_mirror_offset+1 ] = ccf_mirr_i;

    /*
    NOTE: We can get rid of the internal weight-multiplication and apply it instead to
    the references as a pre-process. Here's why that should work:

    ccf_orig += (rr_sr+ri_si) * (i+1);
             += rr_sr*(i+1) + ri_si*(i+1);
             += (rr*sr)*(i+1) + (ri*si)*(i+1);
             += rr*(i+1)*sr + ri*(i+1)*si;)

    -> In each iteration the same constant (i+1) value it multiplied on the same constant
       reference value. We can do that beforehand and avoid four multiplications per loop

    It does not safe that much time though. 
    */
}

// Build cid list in main loop!!!!!!!!!!!!!
__global__ void cu_ccf_mult_m( 
    const float*       sbj_batch_ptr, 
    const float*       ref_batch_ptr, 
    const AlignParam*  u_aln_param,
    float*             batch_table_row_ptr, 
    const unsigned int batch_table_row_offset, 
    const unsigned int batch_table_mirror_offset,
    const unsigned int ring_num,
    const unsigned int refid )
{
    /*
    This kernel is used to compute the cross correlation function of two images, which can be expressed
    as FFT(img_a)' x FFT(img_b). This computation boils down to a number of scalar products with an
    additional factor for weighting the individual rings of the polar representation of the data.
    Note that the function parameters are float* but the values they are pointing to are interpreted as
    complex values -- i.e. every two floats denote a single complex value's real and imaginary parts.
    For details on the cross correlation, see: http://paulbourke.net/miscellaneous/correlate/
    For a walkthrough to the computation, see: https://pastebin.com/zVi13qRZ
    */

    /* Pattern of memory writes

    Each block processes one sbj img and computes its ccf values with the given ref img (the latter is the
    same across all blocks). Note that, at this point, the data is in complex values so the below code
    implements a complex value multiplication.
       We also immediately compute the values for the mirrored subject image. This can be done by multi-
    plying using the complex conjugate (representing a y-axis flipped image). Consequently, each thread
    computes two multiplications and writes two complex result values (four float values).

    Memory layout and access pattern during kernel execution is as follows:

      __ <batch_table_row_ptr> address indicates reference slot (+--+) to be filled in row_0
     /
    V
    [--+--+--+--] .. [--+--+--+--] || [--+--+--+--] .. [--+--+--+--]    First kernel call, given shift_0:
    [--+--+--+--] .. [--+--+--+--] || [--+--+--+--] .. [--+--+--+--]    compute ccf of all sbj imgs and ref_0
    :           :    :           :    :           :                :
    [--+--+--+--] .. [--+--+--+--] || [--+--+--+--] .. [--+--+--+--]

       V
    [==+--+--+--] .. [--+--+--+--] || [==+--+--+--] .. [--+--+--+--]    Second kernel call, given shift_0:
    [==+--+--+--] .. [--+--+--+--] || [==+--+--+--] .. [--+--+--+--]    compute ccf of all sbj imgs and ref_1
    :           :    :           :    :           :                :
    [==+--+--+--] .. [--+--+--+--] || [==+--+--+--] .. [--+--+--+--]

          V
    [==+==+--+--] .. [--+--+--+--] || [==+==+--+--] .. [--+--+--+--]    Third kernel call, given shift_0:
    [==+==+--+--] .. [--+--+--+--] || [==+==+--+--] .. [--+--+--+--]    compute ccf of all sbj imgs and ref_3
    :           :    :           :    :           :                :
    [==+==+--+--] .. [--+--+--+--] || [==+==+--+--] .. [--+--+--+--]

    (..)

    [==+==+==+==] .. [--+--+--+--] || [==+==+==+==] .. [--+--+--+--]    After final kernel call, given shift_0:
    [==+==+==+==] .. [--+--+--+--] || [==+==+==+==] .. [--+--+--+--]    compute ccf of all sbj imgs and ref_N
    :           :    :           :    :           :    :           :
    [==+==+==+==] .. [--+--+--+--] || [==+==+==+==] .. [--+--+--+--]

    (..)

    [==+==+==+==] .. [==+==+==+==] || [==+==+==+==] .. [==+==+==+==]    After final kernel call, given all
    [==+==+==+==] .. [==+==+==+==] || [==+==+==+==] .. [==+==+==+==]    shifts.
    :           :    :           :    :           :                :
    [==+==+==+==] .. [==+==+==+==] || [==+==+==+==] .. [==+==+==+==]

    \__/ blockDim.x*2                              For each given shift, kernel will be called once per
                                                   reference. Afterwards, the batch_table column block for 
    \___________/ blockDim.x*2 * ref_num           the current x/y shift is filled w/ the ccf results of 
                                                   each subject image for all references.
    */

    // grid indices
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;

    // each block picks their reference (NOTE: blockDim.x*2 == ring_len+2)
    //ref_batch_ptr = &ref_batch_ptr[ u_aln_param[bid].ref_id * (blockDim.x*2)*ring_num ];
    ref_batch_ptr = &ref_batch_ptr[ refid * (blockDim.x*2)*ring_num ];

    // each block picks their image (NOTE: blockDim.x*2 == ring_len+2)
    unsigned int sbj_idx = bid * blockDim.x*2*ring_num;  // img_index x img_size

    float ccf_orig_r=0.0f, ccf_orig_i=0.0f, ccf_mirr_r=0.0f, ccf_mirr_i=0.0f;
    for( unsigned int i=0; i<ring_num; i++ ){

        // determine element offsets
        unsigned int ref_elem_idx = i*blockDim.x*2 + tid*2;  // select element in reference image
        unsigned int sbj_elem_idx = sbj_idx + ref_elem_idx;  // select element in subject image
        // read out element data
        float rr = ref_batch_ptr[ ref_elem_idx+0 ];  // ref elem real part
        float ri = ref_batch_ptr[ ref_elem_idx+1 ];  // ref elem imag part
        float sr = sbj_batch_ptr[ sbj_elem_idx+0 ];  // sbj elem real part
        float si = sbj_batch_ptr[ sbj_elem_idx+1 ];  // sbj elem imag part
        // tmps
        float rr_sr = rr * sr;
        float ri_si = ri * si;
        float rr_si = rr * si;
        float ri_sr = ri * sr;
        // weighted ccf values (for original and mirrored sbj image)
        ccf_orig_r += ( rr_sr+ri_si) * (i+1);
        ccf_orig_i += (-rr_si+ri_sr) * (i+1);
        ccf_mirr_r += ( rr_sr-ri_si) * (i+1);
        ccf_mirr_i += (-rr_si-ri_sr) * (i+1);
    }

    // write results
    unsigned int table_idx = blockIdx.x*batch_table_row_offset + tid*2;  // select row and elem_offset
    batch_table_row_ptr[ table_idx+0 ] = ccf_orig_r;
    batch_table_row_ptr[ table_idx+1 ] = ccf_orig_i;
    batch_table_row_ptr[ table_idx+batch_table_mirror_offset+0 ] = ccf_mirr_r;
    batch_table_row_ptr[ table_idx+batch_table_mirror_offset+1 ] = ccf_mirr_i;

    /*
    NOTE: We can get rid of the internal weight-multiplication and apply it instead to
    the references as a pre-process. Here's why that should work:

    ccf_orig += (rr_sr+ri_si) * (i+1);
             += rr_sr*(i+1) + ri_si*(i+1);
             += (rr*sr)*(i+1) + (ri*si)*(i+1);
             += rr*(i+1)*sr + ri*(i+1)*si;)

    -> In each iteration the same constant (i+1) value it multiplied on the same constant
       reference value. We can do that beforehand and avoid four multiplications per loop

    It does not safe that much time though. 
    */
}

__global__ void cu_transform_batch( 
    const cudaTextureObject_t img_src_tex,         // take the img data in this texture
    const unsigned int        img_dim_y,           // images in the texture will have this y-dim
    const AlignParam*         u_aln_param,         // apply these alignment parameters to each img
    float*                    img_storage,         // and store the results in this buffer
    const unsigned int        img_storage_start )  // starting at this index
{

    unsigned int img_idx = blockIdx.x + img_storage_start;  // blockIdx  in [0,img_num_per_texture-1]
    unsigned int img_coord_x = threadIdx.x;                 // threadIdx in [0,img_dim_x-1]
    unsigned int img_storage_idx;

    float img_ctr_x = blockDim.x/2;
    float img_ctr_y = img_dim_y/2;

    float src_coord_x, src_coord_x_tmp, src_coord_y;
    float angle, angle_sin, angle_cos;

    // for each target pixel in the transformed image find the corresponding pixel in the source image
    for( unsigned int img_coord_y=0; img_coord_y<img_dim_y; img_coord_y++ ){

        // mirror
        src_coord_x = (u_aln_param[img_idx].mirror) ? blockDim.x - img_coord_x : img_coord_x;
        src_coord_y = img_coord_y;

        // rotation
        angle = DEG2RAD*u_aln_param[img_idx].angle;
        angle_sin = sinf(angle);
        angle_cos = cosf(angle);

        src_coord_x -= img_ctr_x;
        src_coord_y -= img_ctr_y;

        src_coord_x_tmp = src_coord_x;
        src_coord_x = src_coord_x_tmp*angle_cos - src_coord_y*angle_sin;
        src_coord_y = src_coord_x_tmp*angle_sin + src_coord_y*angle_cos;

        src_coord_x += img_ctr_x;
        src_coord_y += img_ctr_y;

        // shift
        src_coord_x = src_coord_x + u_aln_param[img_idx].shift_x + 0.5;
        src_coord_y = src_coord_y + u_aln_param[img_idx].shift_y + 0.5;
        
        // fetch value from the source image and put into our storage buffer for the transformed data
        img_storage_idx = img_idx * img_dim_y*blockDim.x +  // select image slot
                          img_coord_y * blockDim.x +        // select image row
                          img_coord_x;                      // select image column

        img_storage[ img_storage_idx ] = tex2D<float>( img_src_tex, src_coord_x, blockIdx.x*img_dim_y + src_coord_y );
    }
}

__global__ void cu_average_batch( 
    const float*        img_storage,     // take this buffer holding a bunch of images
    const unsigned int  img_dim_y,       // each of which has this y-dim
    const unsigned int* cid_idx,         // block[i] averages over the images from cid_idx[i] to cid_idx[i+1]
    float**             tgt_tex_data,    // average[i] is written into slot[i] of texture tgt_txt_data[0]
    const size_t        tgt_tex_pitch )  // texture pitch of the allocated 2D memory on the device
{

    unsigned int ref_idx = blockIdx.x;       // [0,ref_num-1]
    unsigned int img_coord_x = threadIdx.x;  // [0,img_dim_x-1]
    unsigned int src_idx, dst_idx;


    for( unsigned int img_coord_y=0; img_coord_y<img_dim_y; img_coord_y++ ){
        float avg = 0.0;
        for( unsigned int img_idx=cid_idx[ref_idx]; img_idx<cid_idx[ref_idx+1]; img_idx++ ){

            src_idx = img_idx * img_dim_y*blockDim.x +
                      img_coord_y * blockDim.x +
                      img_coord_x;

            avg += img_storage[ src_idx ];
        }
        
        dst_idx = ref_idx * img_dim_y*(tgt_tex_pitch/sizeof(float)) +  // NOTE: the incremental unit for pitched
                  img_coord_y * (tgt_tex_pitch/sizeof(float)) +        // memory on the device is a single byte!
                  img_coord_x;

        (tgt_tex_data[0])[dst_idx] = avg / (cid_idx[ref_idx+1]-cid_idx[ref_idx]);
    }
}

__global__ void cu_max_idx_silly(
    const float*       sbj_row_ptr,     // take this buffer holding a number of data rows
    const unsigned int sbj_row_len,     // each row has this length and there is one thread block per row
    int*               sbj_max_idx )    // find the max in each row and store its idx in this array
{

    extern __shared__ float shr_max_val[];

    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;

    unsigned int* shr_max_idx = (unsigned int*)( &shr_max_val[blockDim.x] );
    sbj_row_ptr = &sbj_row_ptr[sbj_row_len*bid];

    float max_val = 0.0f;
    int   max_idx = 0;

    unsigned int start_idx = (sbj_row_len/blockDim.x + 1) * (tid+0);
    unsigned int stop_idx  = (sbj_row_len/blockDim.x + 1) * (tid+1);
    stop_idx = ( stop_idx > sbj_row_len ) ? sbj_row_len : stop_idx;

    for( unsigned int idx=start_idx; idx<stop_idx; idx++ ){
        if( sbj_row_ptr[idx] > max_val ){
            max_val = sbj_row_ptr[idx];
            max_idx = idx;
        }
    }
    shr_max_val[tid] = max_val;
    shr_max_idx[tid] = max_idx;
    
    __syncthreads();

    if( tid == 0 ){
        for( unsigned int i=0; i<blockDim.x; i++ ){
            if( shr_max_val[i] > max_val ){
                max_val = shr_max_val[i];
                max_idx = shr_max_idx[i];
            }
        }
        sbj_max_idx[bid] = max_idx;
    }
}

//=======================================================[ BatchHandler class ]

BatchHandler::BatchHandler( const AlignConfig* batch_cfg, const char* batch_type ){

    // sanity check
    assert( strcmp(batch_type, "subject_batch")==0       || 
            strcmp(batch_type, "reference_batch")==0     ||
            strcmp(batch_type, "pre_align_ref_batch")==0 );
    this->batch_type = batch_type;

    // img params
    if( strcmp(batch_type, "subject_batch") == 0 )
        img_num = batch_cfg->sbj_num;
    else
        img_num = batch_cfg->ref_num;

    img_dim_x = batch_cfg->img_dim;
    img_dim_y = batch_cfg->img_dim;

    // determine number of textures needed to hold all data
    cudaDeviceProp cu_dev_prop;
    CUDA_ERR_CHK( cudaGetDeviceProperties(&cu_dev_prop, CUDA_DEVICE_ID) );

    img_num_per_tex = cu_dev_prop.maxTexture2DLinear[1] / img_dim_x;  // max. 2D texture height in linear memory by img_dim_x
    img_tex_num     = (img_num%img_num_per_tex == 0) ? img_num/img_num_per_tex : img_num/img_num_per_tex + 1;  // number of texture objects
    img_tex_obj     = (cudaTextureObject_t*)malloc( img_tex_num*sizeof(cudaTextureObject_t) );                 // array of texture objects
    CUDA_ERR_CHK( cudaMallocManaged(&u_img_tex_data, img_tex_num*sizeof(float*)) );

    if( strcmp(batch_type, "reference_batch")==0 || strcmp(batch_type, "pre_align_ref_batch")==0 ){
        if( img_num > img_num_per_tex ){
            printf( "\nERROR! BatchHandler::BatchHandler(): img_num[%d] > img_num_per_tex[%d] for type \'%s\'\n\n", img_num, img_num_per_tex, batch_type );
            assert( img_num <= img_num_per_tex ); // see BatchHandler::fetch_averages() for why this is an issue
        }
    }

    // allocate texture buffers
    for( unsigned int i=0; i<img_tex_num; i++ ){
        unsigned int allocate_img_num = (i!=img_tex_num-1) ? img_num_per_tex : img_num - i*img_num_per_tex;
        create_texture_objects( allocate_img_num, i );
    }

    // polar data params & buffer allocation
    this->ring_num = batch_cfg->ring_num;
    this->ring_len = batch_cfg->ring_len;

    /* NOTE: The polar representation has width <ring_len> BUT will later be
    subject to an in-place FFT operation which will require a width of 
    <ring_len+2> so we allocate the full memory right away. This is also why
    the polar resampling code operates on  <ring_len> values but uses 
    <ring_len+2> for indexing.

       Further, note that the polar data consists of floats, while the FFT'd
    data consists of complex values (i.e. pairs of float). This means that the
    "+2" in <ring_len+2> provides enough space to hold ONE additional complex 
    number, and the overall number of complex values is <ring_len/2+1>. 

       NOTE: The d_img_data buffer is used to hold the image data (a) after
    polar conversion (as described above), and (b) after applying the last
    computed alignment parameters. To make sure the buffer is able to hold
    either of the two we allocate the max() memory requirement of the two. */

    unsigned int polar_buffer_size   = img_num*(ring_len+2)*ring_num*sizeof(float);
    //unsigned int trans_buffer_size = img_num* img_dim_x   *img_dim_y*sizeof(float);  // always smaller than the one below; only here as a reminder
    unsigned int   fft_buffer_size   = img_num*(img_dim_x+2)*img_dim_y*sizeof(float);  // in-place fft needs one additional column of complex values

    //CUDA_ERR_CHK ( cudaMallocManaged(&d_img_data, max(polar_buffer_size+1024*1024, fft_buffer_size+1024*1024)) );
    CUDA_ERR_CHK ( cudaMallocManaged(&d_img_data, max(polar_buffer_size, fft_buffer_size)) );

    // CUFFT plan
    CUFFT_ERR_CHK( cufftPlan1d(&cufft_pln, ring_len, CUFFT_R2C, ring_num*img_num) );

    // subject batch handler: create a ccf table
    if( strcmp(batch_type, "subject_batch") == 0 )
        ccf_table = new CcfResultTable( batch_cfg );
    else
        ccf_table = NULL;

    // reference batch handler: create CUFFT plans to be used in the tangent filter
    if( strcmp(batch_type, "reference_batch") == 0 ){

        int n[] = { static_cast<int>(img_dim_y), static_cast<int>(img_dim_x) };

        int idist = img_dim_y * (static_cast<int>(img_tex_pitch)/sizeof(float));
        int odist = img_dim_y * (static_cast<int>(img_tex_pitch)/sizeof(float2));

        int inembed[] = { static_cast<int>(img_dim_y), (static_cast<int>(img_tex_pitch/sizeof(float ))) };
        int onembed[] = { static_cast<int>(img_dim_y), (static_cast<int>(img_tex_pitch/sizeof(float2))) };

        // NOTE: img_tex_pitch is defined in units of bytes, whereas advanced
        //       memory layout parameters are in units of the relevant data type.
        // NOTE: Advanced data layout parameters doc:
        //       - https://docs.nvidia.com/cuda/cufft/index.html#advanced-data-layout
        //       - https://stackoverflow.com/questions/22953171/batched-ffts-using-cufftplanmany#23036876

        CUFFT_ERR_CHK( cufftPlanMany( 
            &cufft_pln_filter_in, 2, n,
            inembed, 1, idist,
            onembed, 1, odist,
            CUFFT_R2C, img_num));

        CUFFT_ERR_CHK( cufftPlanMany(
            &cufft_pln_filter_out, 2, n,
            onembed, 1, odist,
            inembed, 1, idist,
            CUFFT_C2R, img_num));
    }
    else
        cufft_pln_filter_in = cufft_pln_filter_out = 0;
}

BatchHandler::BatchHandler(){
    // data param
    img_num = img_dim_x = img_dim_y = 0;
    // ccf table
    ccf_table = NULL;
    // cuda handles
    cufft_pln = 0;
    // texture data
    img_tex_num = img_num_per_tex = 0;
    u_img_tex_data = NULL;
    img_tex_obj = NULL;
    // polar/FFT param
    ring_num = ring_len = 0;
    // processed image data
    d_img_data = NULL;
}

BatchHandler::~BatchHandler(){
    // ccf tbale
    if( ccf_table != NULL ){
        delete( ccf_table );
        ccf_table = NULL;
    }
    // destroy raw data behind the texture objects
    if( u_img_tex_data != NULL ){
        for( unsigned int i=0; i<img_tex_num; i++ )
            CUDA_ERR_CHK( cudaFree(u_img_tex_data[i]) );
        cudaFree(u_img_tex_data);
        u_img_tex_data = NULL;
    }
    // destroy texture objects
    if( img_tex_obj != NULL ){
        for( unsigned int i=0; i<img_tex_num; i++ )
            CUDA_ERR_CHK( cudaDestroyTextureObject(img_tex_obj[i]) );
        img_num_per_tex = 0;
        img_tex_obj = NULL;
        img_tex_num = 0;
    }
    // destroy buffer for processed image data
    if( d_img_data != NULL ){
        CUDA_ERR_CHK( cudaFree(d_img_data) );
        d_img_data = NULL;
    }
    // destroy cufft plans
    if( cufft_pln != 0 ){
        CUFFT_ERR_CHK( cufftDestroy(cufft_pln) );
        cufft_pln = 0;
    }
    if( cufft_pln_filter_in != 0 ){
        CUFFT_ERR_CHK( cufftDestroy(cufft_pln_filter_in) );
        cufft_pln_filter_in = 0;
    }
    if( cufft_pln_filter_out != 0 ){
        CUFFT_ERR_CHK( cufftDestroy(cufft_pln_filter_out) );
        cufft_pln_filter_out = 0;
    }
}

//---------------------------------------------------------[ public interface ]

unsigned int BatchHandler::size() const { return img_num; }

array<unsigned int,2> BatchHandler::ring_param() const{
    array<unsigned int,2> param = { ring_num, ring_len };
    return param;
}

void BatchHandler::fetch_data(
    const float**      img_data,
    const unsigned int img_idx,
    const unsigned int img_limit )
{
    /*
    Fetch data from img_data array. Data will be read starting at index img_idx
    and we fetch as much data as we can fit in the available texture buffers.

    NOTE: At this point we assume that the outer Python code makes sure to call
    the alignment only with sizes that will fit into GPU memory.

    Args:
        img_data:  Image data array.
        img_idx:   Start copying at this image in the array.
        img_limit: This is the amount of total images in the array. We assume
                   that the outside Python code doesn't send us more data than
                   we can handle and we copy everything into our buffers.
    */

    // make sure we're not fetching more images than we can handle in our texture buffers
    assert( img_limit <= img_tex_num*img_num_per_tex );

    // update number of images stored after data fetch
    img_num = img_limit;

    // go fetch
    for( unsigned int tex_idx=0; tex_idx < img_tex_num; tex_idx++ ){
        // number of images to be copied into the selected texture and starting index of the first image to be copied
        unsigned int move_img_num = (tex_idx != img_tex_num-1) ? img_num_per_tex : img_num - tex_idx*img_num_per_tex;
        unsigned int move_img_idx = img_idx + tex_idx*img_num_per_tex;

        // remaining data only fills part of the next texture buffer; this is
        // normal and happens whenever we fill the final texture buffer
        if( move_img_idx + move_img_num > img_num )
            move_img_num = img_num - move_img_idx;

        // if we have copied all data but at least one texture buffer is still
        // empty, we break; this only happens when the BatchHandler has to deal
        // with batches of different sizes
        if( move_img_idx >= img_num ) break;

        // copy the data from individual images into one of several allocated continuous texture memory buffers
        unsigned int offset = 0;
        for( unsigned int i=move_img_idx; i < move_img_idx + move_img_num; i++ ){
            CUDA_ERR_CHK( cudaMemcpy2D( &(u_img_tex_data[tex_idx])[offset], img_tex_pitch,  // destination address and pitch
                                        img_data[i], img_dim_x*sizeof(float),               // source address and pitch
                                        img_dim_x*sizeof(float), img_dim_y,                 // data transfer width and height
                                        cudaMemcpyHostToDevice ) );
            offset += img_dim_y * img_tex_pitch / sizeof(float);  // <img_tex_pitch> is in units of bytes but when used for indexing it will be assumed to be in units of sizeof(float)
        }
    }
}

void BatchHandler::resample_to_polar( 
    const float        shift_x, 
    const float        shift_y, 
    const unsigned int data_idx,
    const float*       u_polar_sample_coords )
{
    unsigned int img_data_idx, img_idx=0;
    for( unsigned int i=0; i<img_tex_num; i++ ){

        // gefahr: final texture buffer is probably not filled completely
        unsigned int tmp_img_num = (i!=img_tex_num-1) ? img_num_per_tex : img_num - i*img_num_per_tex;

        // gefahr: we may have fetched a work load that doesn't require all texture buffers
        if( img_idx + tmp_img_num > img_num )
            tmp_img_num = img_num - img_idx;

        dim3 grid_dim(tmp_img_num, ring_num);
        img_data_idx  = img_idx * (ring_len+2)*ring_num;
        AlignParam* aln_param = (ccf_table != NULL) ? &aln_res.u_aln_param[data_idx + img_idx] : NULL;

        cu_resample_to_polar<<< grid_dim, ring_len >>>(
            img_tex_obj[i],               // IN:  data held by texture object[i]
            aln_param,                    // IN:  accumulated shift as aln_param[i].shift_x/y
            &d_img_data[img_data_idx],    // OUT: polar data at given idx
            img_dim_x, img_dim_y,         // CONST: img dimensions
            shift_x, shift_y,             // PARAM: shift parameters
            u_polar_sample_coords,        // CONST: resampling coordinates (constants)
            ring_len, ring_num);          // CONST: resampling parameters (constants)

        KERNEL_ERR_CHK();

        // break once we haven't processed a full texture (regardless of whether or not that texture is the final one)
        if( tmp_img_num != img_num_per_tex )
            break;

        // otherwise move on to the next texture buffer and process its contents
        else
            img_idx += tmp_img_num;
    }
}

void BatchHandler::apply_FFT(){
    assert( d_img_data != NULL );
    CUFFT_ERR_CHK( cufftExecR2C(cufft_pln, (cufftReal*)d_img_data, (cufftComplex*)d_img_data) );
    CUDA_ERR_CHK( cudaDeviceSynchronize() );
}

void BatchHandler::ccf_mult(
    const BatchHandler* ref_batch,
    const unsigned int  shift_idx,
    const unsigned int  data_idx )
{
    // invoke cuda kernel to process the ccf multiplications
    cu_ccf_mult<<< img_num, ccf_table->get_ring_len()/2+1 >>>(
        d_img_data,                        // IN: take all our images and the selected reference
        ref_batch->img_ptr(0),             // IN: ...
        &aln_res.u_aln_param[data_idx],    // IN: sbj_cid in form of aln_param[i].ref_id
        ccf_table->row_ptr(shift_idx, 0),  // OUT: in-row offset for results of given shift and reference, reference 0~n
        ccf_table->row_off(),              // CONST: offset to reach successive rows
        ccf_table->mirror_off(),           // CONST: in-row offset for mirrored results
        ring_num);                         // CONST: polar sampling parameters (ring length)
    KERNEL_ERR_CHK();
}

void BatchHandler::ccf_mult_m(
    const BatchHandler* ref_batch,
    const unsigned int  shift_idx,
    const unsigned int  data_idx )
{
    // invoke cuda kernel to process the ccf multiplications, this will go through all the reference
	for ( unsigned int j=0; j<ref_batch->img_num; j++ ){
        cu_ccf_mult_m<<< img_num, ccf_table->get_ring_len()/2+1 >>>(
            d_img_data,                        // IN: take all our images and the selected reference
            ref_batch->img_ptr(j),             // IN: ...
            &aln_res.u_aln_param[data_idx],    // IN: sbj_cid in form of aln_param[i].ref_id, no used here
            ccf_table->row_ptr(shift_idx, j),  // OUT: in-row offset for results of given shift and reference, reference 0~n
            ccf_table->row_off(),              // CONST: offset to reach successive rows
            ccf_table->mirror_off(),           // CONST: in-row offset for mirrored results
            ring_num,
            j);                         // CONST: polar sampling parameters (ring length)
        KERNEL_ERR_CHK();
	}
}

void BatchHandler::apply_IFFT(){ ccf_table->apply_IFFT(); }

void BatchHandler::apply_alignment_param( AlignParam* aln_param ){
    
    unsigned int img_storage_idx=0;
    for( unsigned int i=0; i<img_tex_num; i++ ){

        unsigned int tmp_img_num = (i!=img_tex_num-1) ? img_num_per_tex : img_num - i*img_num_per_tex;

        cu_transform_batch<<< tmp_img_num, img_dim_x >>>(
            img_tex_obj[i],
            img_dim_y,
            aln_param,
            d_img_data,
            img_storage_idx);

        KERNEL_ERR_CHK();
        img_storage_idx += tmp_img_num;
    }
    CUDA_ERR_CHK( cudaDeviceSynchronize() );
}

void BatchHandler::fetch_averages( float* img_data ){

    /* WARNING: This assumes that all references fit into a single texture! */

    cu_average_batch<<< img_num, img_dim_x >>>(
        img_data,
        img_dim_y,
        aln_res.u_cid_idx,
        u_img_tex_data,
        img_tex_pitch);

    KERNEL_ERR_CHK();
    CUDA_ERR_CHK( cudaDeviceSynchronize() );
}

void BatchHandler::apply_tangent_filter( const float cutoff_freq, const float falloff ){

    /* Applies a tangent filter to the images stored in the TEXTURE memory
    buffer of the BatchHandler instance. Note that the texture buffer stores 
    real images. Since the tangent filter operates in frequency space we need
    to sandwich the actual filter operation in between two FFTs. Further, to
    not mess up our image data we also need to normalize our data after each
    FFT call since CUDA performs a non-normalizing FFT.

       For more information see:
    http://sparx-em.org/sparxwiki/filt_tanl
    http://sparx-em.org/sparxwiki/absolute_frequency_units
    http://www.balkangeophysoc.gr/online-journal/1998_V1/feb1998/PDF/Feb98-14.pdf
    https://groups.google.com/forum/#!topic/eman2/a1SDCNggmOA

    Args:

        cutoff_freq (float): Filter frequency in [0.0, 0.5] (higher values mean
            more frequencies are retained, resulting in a sharper image).

        fallof (float): Stepness of the filter in [0.0, 0.5] (lower values mean
            less transition space, resulting in a steeper transition).
    */

    /* WARNING: This assumes that all references fit into a single texture! */
    /*          -> Normalization only copies from/to a single texture.      */
    /*          -> Tangent filter kernel only operates on a single texture. */


    // STEP 01: apply FFT to images (and normalize manually)
    CUFFT_ERR_CHK( cufftExecR2C(cufft_pln_filter_in, (cufftReal*)u_img_tex_data[0], (cufftComplex*)u_img_tex_data[0]) );

    float tmp_space[img_dim_y*(img_tex_pitch/sizeof(float))];
    CUDA_ERR_CHK( cudaMemcpy(tmp_space, u_img_tex_data[0], img_dim_y*(img_tex_pitch/sizeof(float)) *sizeof(float), cudaMemcpyDeviceToHost) );
    for( unsigned int y=0; y<img_dim_y; y++){
        for( unsigned int x=0; x<img_tex_pitch/sizeof(float); x++)
            tmp_space[y*(img_tex_pitch/sizeof(float))+x] /= img_dim_x*img_dim_y;
    }
    CUDA_ERR_CHK( cudaMemcpy(u_img_tex_data[0], tmp_space, img_dim_y*(img_tex_pitch/sizeof(float)) *sizeof(float), cudaMemcpyHostToDevice) );

    // STEP 02: run the low-pass tangent filter
    cu_apply_tanl_filter_to_tex<<< img_num, img_dim_x/2+1 >>>(
        (cuComplex*)u_img_tex_data[0],
        img_tex_pitch,
        img_dim_y,
        cutoff_freq,
        falloff);
    
    KERNEL_ERR_CHK();
    CUDA_ERR_CHK( cudaDeviceSynchronize() );

    // STEP 03: apply IFFT to images (and normalize manually)
    CUFFT_ERR_CHK( cufftExecC2R(cufft_pln_filter_out, (cufftComplex*)u_img_tex_data[0], (cufftReal*)u_img_tex_data[0]) );

    CUDA_ERR_CHK( cudaMemcpy(tmp_space, u_img_tex_data[0], img_dim_y*(img_tex_pitch/sizeof(float)) *sizeof(float), cudaMemcpyDeviceToHost) );
    for( unsigned int y=0; y<img_dim_y; y++){
        for( unsigned int x=0; x<img_tex_pitch/sizeof(float); x++)
            tmp_space[y*(img_tex_pitch/sizeof(float))+x] /= img_dim_x*img_dim_y;
    }
    CUDA_ERR_CHK( cudaMemcpy(u_img_tex_data[0], tmp_space, img_dim_y*(img_tex_pitch/sizeof(float)) *sizeof(float), cudaMemcpyHostToDevice) );

    // this can catch errors that fall through otherwise, so better safe than sorry here
    CUDA_ERR_CHK( cudaDeviceSynchronize() );
}

void BatchHandler::compute_alignment_param( 
    const unsigned int            param_idx, 
    const unsigned int            param_limit, 
    const vector<array<float,2>>* shifts,
    AlignParam*                   aln_param )
{
    ccf_table->compute_alignment_param( param_idx, param_limit, shifts, aln_param );
}

float* BatchHandler::img_ptr( const unsigned int img_idx ) const {
    // return pointer to img_i within the continuous stack data array
    assert( d_img_data != NULL );
    assert( img_idx < img_num );
    return &d_img_data[ img_idx*(ring_len+2)*ring_num ];
}

//-----------------------------------------------------------------[ privates ]

void BatchHandler::create_texture_objects( const unsigned int num, const unsigned int tex_obj_idx ){
    // allocate pitched memory on the device
    CUDA_ERR_CHK( cudaMallocPitch( &(u_img_tex_data[tex_obj_idx]), &img_tex_pitch, img_dim_x*sizeof(float), img_dim_y*num ) );

    // generic resource descriptor
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypePitch2D;
    res_desc.res.pitch2D.desc   = cudaCreateChannelDesc<float>();
    res_desc.res.pitch2D.devPtr = u_img_tex_data[tex_obj_idx];
    res_desc.res.pitch2D.width  = img_dim_x;
    res_desc.res.pitch2D.height = img_dim_y*num;
    res_desc.res.pitch2D.pitchInBytes = img_tex_pitch;

    // texture descriptor
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode     = cudaFilterModeLinear;  // for linear interpolation (NOTE: this breaks normal integer indexing!)
    tex_desc.readMode       = cudaReadModeElementType;
    tex_desc.normalizedCoords = false;  // we want to index using [0;img_dim] rather than the texture default of [0;1]

    // create texture object
    CUDA_ERR_CHK( cudaCreateTextureObject(&img_tex_obj[tex_obj_idx], &res_desc, &tex_desc, NULL) );
}

//======================================================[ CcfResultTable class]

CcfResultTable::CcfResultTable( const AlignConfig* batch_cfg ){
    /*
    This table holds the information of cross-correlating all (mirrored) subjects with all images
    given all shifts. All information is stored in a single, continuous array that can be regarded
    as a 2D matrix with each row storing all ccf information for one subject (per reference, per
    shift, per original/mirrored).
       This isn't necessarily pretty but it works wel with CUDA kernels that like to work on
    continuous data ranges. Within each row, the data is organized as follows:

                               __..................... ccf_mult result for one sbj and one ref image
                              /  \          
    [--+--+--+--] .. [--+--+--+--] || [--+--+--+--] .. [--+--+--+--] ....... one row per subject img
    [--+--+--+--]    \___________/
         ...               ^............... ccf result for one sbj and all ref images w/ given shift
    [--+--+--+--]     
                                      \____________________________/
                                                     ^............. ccf result for mirrored sbj imgs

    NOTE: "ccf result" refers to the result of IFFT( FFT(sbj_i)' x FFT(ref_j) ), i.e., the cross-
    correlation of subject image [i] and reference image [j]. one ccf result slot holds <ring_len+2> 
    values.
    */

    // get parameters
    sbj_num = batch_cfg->sbj_num;
    ref_num = 1; // for reference free alignment each sbj img only compares with its own, single, reference

    ring_num = batch_cfg->ring_num;
    ring_len = batch_cfg->ring_len;

    assert( fmod(batch_cfg->shift_rng_x, batch_cfg->shift_step) == 0 );
    assert( fmod(batch_cfg->shift_rng_y, batch_cfg->shift_step) == 0 );
    shift_num = (2*(batch_cfg->shift_rng_x/batch_cfg->shift_step)+1) * (2*(batch_cfg->shift_rng_y/batch_cfg->shift_step)+1);

    // ccf_mult result of len (ring_len+2) * per sbj * per ref * per shift * 2 for mirrored sbj images
    size_t batch_table_size = (ring_len+2) * sbj_num * ref_num * shift_num * 2;
    CUDA_ERR_CHK( cudaMallocManaged(&u_ccf_batch_table, batch_table_size*sizeof(float)) );

    // allocate space for the results of reverse engineering the alignment parameters per sbj img
    u_max_idx=NULL;
    CUDA_ERR_CHK( cudaMallocManaged(&u_max_idx, sbj_num*sizeof(unsigned int)) );

    // cuda handles
    CUFFT_ERR_CHK( cufftPlan1d(&cufft_pln, ring_len, CUFFT_C2R, entry_num()) );
}

CcfResultTable::CcfResultTable(){
    sbj_num = ref_num = ring_num = ring_len = shift_num = 0;
    u_ccf_batch_table = NULL;
    u_max_idx = NULL;
    cufft_pln = 0;
}

CcfResultTable::~CcfResultTable(){
    if( u_ccf_batch_table != NULL ){
        CUDA_ERR_CHK( cudaFree(u_ccf_batch_table) );
        u_ccf_batch_table = NULL;
    }
    if( u_max_idx != NULL ){
        CUDA_ERR_CHK( cudaFree(u_max_idx) );
        u_max_idx = NULL;
    }
    if( cufft_pln != 0 ){
        CUFFT_ERR_CHK( cufftDestroy(cufft_pln) );
        cufft_pln = 0;
    }
}

//---------------------------------------------------------[ public interface ]

/*
The following *_off() functions return the relevant offsets to navigate the 
continuous data array at the heart of the CcfResultTable class, i.e., to skip
rows (row_off()), original/mirrored half of a row (mirror_off()), shift blocks 
(shift_off), and ccf results of individual sbj-ref-pairs (ref_off()).
*/

inline unsigned int CcfResultTable::row_off() const { return (ring_len+2)*ref_num*shift_num*2; }

inline unsigned int CcfResultTable::mirror_off() const { return (ring_len+2)*ref_num*shift_num; }

inline unsigned int CcfResultTable::shift_off() const { return (ring_len+2)*ref_num; }

inline unsigned int CcfResultTable::ref_off() const { return (ring_len+2); }

float* CcfResultTable::row_ptr( const unsigned int shift_idx, const unsigned int ref_idx ) const {
    // Returns pointer to shift block <shift_idx> at ccf result w/ reference <ref_idx> in row_0.
    // NOTE: To get the same for row_i we can add row_ptr() + i*row_off().
    assert( ref_idx < ref_num );
    assert( shift_idx < shift_num );
    return &u_ccf_batch_table[ shift_off()*shift_idx + ref_off()*ref_idx ];
}

inline unsigned int CcfResultTable::get_ring_num() const{ return ring_num; }

inline unsigned int CcfResultTable::get_ring_len() const{ return ring_len; }

inline unsigned int CcfResultTable::row_num() const { return sbj_num; }

inline unsigned int CcfResultTable::entry_num() const { return sbj_num * ref_num * shift_num * 2; }

inline size_t CcfResultTable::memsize() const { return (ring_len+2) * entry_num() * sizeof(float); }

void CcfResultTable::apply_IFFT(){
    /*
    For each ccf result entry this reads <ring_len/2> complex values to produce <ring_len>
    float values. It is the inverse operation to BatchHandler::applyFFT(). For more 
    information see the doc in BatchTable::apply_FFT() and BatchTable::apply_IFFT().
    */
    CUFFT_ERR_CHK( cufftExecC2R(cufft_pln, (cufftComplex*)u_ccf_batch_table, (cufftReal*)u_ccf_batch_table) );
    CUDA_ERR_CHK ( cudaDeviceSynchronize() );
}

void CcfResultTable::compute_alignment_param(
    const unsigned int            param_idx,
    const unsigned int            param_limit,
    const vector<array<float,2>>* shifts,
    AlignParam*                   aln_param )
{
    // prep
    float shift_limit = aln_res.aln_cfg->img_dim - aln_res.aln_cfg->ring_num - 2;

    // step 01: compute idx of max within each row of the ccf table
    compute_max_indices();

    // step 02: reverse engineer alignment parameters for sbj_i given max idx value within data row_i
    for( unsigned int i=0; i<sbj_num; i++ ){
        // did we run out of data? (this only happens once at the very end)
        if( param_idx + i >= param_limit )
            break;
        // next parameter set
        AlignParam* tmp_param = &aln_param[param_idx + i];
        int idx = u_max_idx[i];
        // mirror flag
        if( idx >= mirror_off() ){ tmp_param->mirror = true; idx -= mirror_off(); }
        else tmp_param->mirror = false;
        // shifts
        tmp_param->shift_x += (*shifts)[idx/shift_off()][0];
        tmp_param->shift_y += (*shifts)[idx/shift_off()][1];
        tmp_param->shift_x  = min(max(tmp_param->shift_x, -shift_limit), shift_limit);
        tmp_param->shift_y  = min(max(tmp_param->shift_y, -shift_limit), shift_limit);
        idx -= shift_off()*(idx/shift_off());
        // ref img id
        int ref_id = idx/ref_off();
        idx -= ref_id*ref_off();
        // rotation angle (interpolated & adjusted for EMAN2 compatibility)
        tmp_param->angle = 360.0 - interpolate_angle( i, u_max_idx[i], idx );
        if( tmp_param->mirror ){
            tmp_param->angle += 180.0;
            if( tmp_param->angle >= 360.0 )
                tmp_param->angle -= 360.0;
        }
    }
}

//-----------------------------------------------------------------[ privates ]

void CcfResultTable::compute_max_indices() {
    unsigned int threads = 128;
    cu_max_idx_silly<<< sbj_num, threads, (threads*sizeof(float)+threads*sizeof(unsigned int)) >>>( u_ccf_batch_table, row_off(), u_max_idx );
    KERNEL_ERR_CHK();
    CUDA_ERR_CHK( cudaDeviceSynchronize() );
}

double CcfResultTable::interpolate_angle(
    const unsigned int sbj_idx,
    const unsigned int max_idx,
    const unsigned int max_idx_off )
{
    /*
    ccf table:
    sbj_idx-1:  .. [----][----][----]..
    sbj_idx     .. [----][--x-][----]..  <-- x marks the peak value found at ccf_table[ sbj_idx*row_off() + max_idx ]
    sbj_idx+1:  .. [----][----][----]..
                               \____/_______ Each [----] block holds the ccf results for all angles given a specific reference
    Args:                                                                                                            and shift
        unsigned int sbj_idx: Index of the sbj image that we're rotating atm. Used to find the 
            correct row in the ccf results table.

        unsigned int max_idx: Index of the maximum within the sbj row (within [0,row_off]).

        unsigned int max_idx_off: Offset of the <max_idx> value from the beginning of its ccf-
            result block:             __ _______________
                                     |  |               \___ <max_idx_off> (within [0,ring_len+2])
            row[ sbj_idx ]: .. [----][--x-][----]..
    */

    // collect cross reference values around peak angle
    double x[7];
    for( int i=-3; i<=3; i++ ){

        unsigned int base = 0;                  // pointer to beginning of the ccf result that contains the peak value, like so:
        base += sbj_idx*row_off();              // select the row for the subject in question
        base += (max_idx/ref_off())*ref_off();  // select the result; <max_idx_off> is now the offset from <base> to the peak angle

        x[i+3] = u_ccf_batch_table[ base + (max_idx_off+i)%ring_len ];  // NOTE: not (ring_len+2) !
    }

    // fit parabola to the collected values and find the location of the parabola's peak
    // NOTE: parabolic fit adapted from the sparx Util::prb1d() function
    double c2 =  49.*x[0] + 6.*x[1] - 21.*x[2] - 32.*x[3] - 27.*x[4] - 6.*x[5] + 31.*x[6];
    double c3 =   5.*x[0] - 3.*x[2] -  4.*x[3] -  3.*x[4] +  5.*x[6];

    // interpolate the angle in between the discrete angular steps of the alignment search
    double angle_step = (360.0/(double)(ring_len));
    double angle = angle_step * (double)(max_idx_off);  // NOTE: this would be our estimated angle w/o interpolation

    if( c3 != 0.0 )
        return angle + angle_step*( c2 / (2.0*c3) - 4 );  // term in brackets is the interpolation factor
    else
        return angle;
}

//==================================================================[ testing ]

//--------------------------------------------------[ pre_align_run() testing ]

float** create_rnd_data( const unsigned int img_num, const unsigned img_dim ){
    float** img_data = (float**)malloc( img_num*img_dim*sizeof(float*) );
    for( unsigned int i=0; i<img_num; i++ ){
        for( unsigned int y=0; y<img_dim; y++ ){
            img_data[img_dim*i+y] = (float*)malloc(img_dim*sizeof(float));
            for( unsigned int x=0; x<img_dim; x++)
                img_data[img_dim*i+y][x] = rand()%255;
        }
    }
    return img_data;
}

void print_data( const float** img_data, const unsigned int img_num, const unsigned int img_dim ){
    for(int i=0; i<img_num; i++){
        printf( "img[%d]:\n", i );
        for( unsigned int y=0; y<img_dim; y++ ){
            for( unsigned int x=0; x<img_dim; x++)
                printf("%.2f ", img_data[img_dim*i+y][x]);
            printf("\n");
        }
    }
}

void delete_rnd_data( float** img_data, const unsigned int img_num, const unsigned int img_dim ){
    for( unsigned int i=0; i<img_num; i++ ){
        for( unsigned int y=0; y<img_dim; y++ )
            delete( img_data[img_dim*i+y] );
    }
    delete( img_data );
}

size_t gpu_mem_aloc(){
    size_t mem_avl=0, mem_ttl=0;
    CUDA_ERR_CHK( cudaDeviceSynchronize() );
    CUDA_ERR_CHK( cudaMemGetInfo(&mem_avl, &mem_ttl) );
    return (mem_ttl/(1024*1024) - mem_avl/(1024*1024));
}

//--------------------------------------------------------------[ test archive ]

// size estimate debugging
int main_2( int argc, const char** argv ){

    // gpu init
    CUDA_DEVICE_ID = 0;
    printf( "set GPU[%d]\n", CUDA_DEVICE_ID );
    CUDA_ERR_CHK( cudaSetDevice(CUDA_DEVICE_ID) );
    print_gpu_info(CUDA_DEVICE_ID);
    printf("\n");

    // set up that shit
    unsigned int n = 50000;

    // param
    AlignConfig aln_cfg = AlignConfig();
    aln_cfg.sbj_num     =   n;
    aln_cfg.ref_num     =   1;
    aln_cfg.img_dim     =  76;
    aln_cfg.ring_num    =  29;
    aln_cfg.ring_len    = 256;
    aln_cfg.shift_step  =   1;
    aln_cfg.shift_rng_x =   3;
    aln_cfg.shift_rng_y =   3;

    // mem estimate
    pre_align_size_check( n, &aln_cfg, CUDA_DEVICE_ID, 0.9, true );
    CUDA_ERR_CHK( cudaDeviceSynchronize() );
    printf("\n");

    // init
    printf( "MEM PRE  INIT: %zu MB\n", gpu_mem_aloc() );
    AlignParam* aln_param = pre_align_init( aln_cfg.sbj_num, &aln_cfg, CUDA_DEVICE_ID );
    printf( "MEM POST INIT: %zu MB\n", gpu_mem_aloc() );

    // create & fetch sbj data
    printf( "create sbj: %ux rnd img w/ dim(%u,%u).. ", aln_cfg.sbj_num, aln_cfg.img_dim, aln_cfg.img_dim );
    float** sbj_data = create_rnd_data( aln_cfg.sbj_num, aln_cfg.img_dim );
    pre_align_fetch( (const float**)sbj_data, aln_cfg.sbj_num, "sbj_batch" );
    printf("done\n");

    // create & fetch ref data
    printf( "create ref: %dx rnd img data w/ dim(%d,%d).. ", aln_cfg.ref_num, aln_cfg.img_dim, aln_cfg.img_dim );
    float** ref_data = create_rnd_data( aln_cfg.ref_num, aln_cfg.img_dim );
    pre_align_fetch( (const float**)ref_data, aln_cfg.ref_num, "ref_batch" );
    printf("done\n");

    // alignment call
    printf( "MEM PRE  ALIGN: %zu MB\n", gpu_mem_aloc() );
    pre_align_run( 0, aln_cfg.sbj_num );
    printf( "MEM POST ALIGN: %zu MB\n", gpu_mem_aloc() );

    // clean up
    printf( "MEM PRE  CLEAR: %zu MB\n", gpu_mem_aloc() );
    gpu_clear();
    printf( "MEM POST CLEAR: %zu MB\n", gpu_mem_aloc() );

    // exit
    printf( "\nall done\n" );
    return EXIT_SUCCESS;
}

// pre-align
int main_1( int argc, const char** argv ){

    // gpu init
    CUDA_DEVICE_ID = 0;
    printf( "set GPU[%d]\n", CUDA_DEVICE_ID );
    CUDA_ERR_CHK( cudaSetDevice(CUDA_DEVICE_ID) );
    print_gpu_info(CUDA_DEVICE_ID);

    // param
    AlignConfig aln_cfg = AlignConfig();
    aln_cfg.sbj_num     = 10000;
    aln_cfg.ref_num     =     1;
    aln_cfg.img_dim     =    80;
    aln_cfg.ring_num    =    29;
    aln_cfg.ring_len    =   256;
    aln_cfg.shift_step  =     2;
    aln_cfg.shift_rng_x =     8;
    aln_cfg.shift_rng_y =     8;

    // aln init
    AlignParam* aln_param = pre_align_init( aln_cfg.sbj_num, &aln_cfg, CUDA_DEVICE_ID );

    // shift array
    printf( "len shift array: %zu\n", aln_res.shifts->size() );
    //for( auto &s : *(aln_res.shifts) ) printf( "%d, %d\n", s[0], s[1] );

    // sbj data
    printf( "create sbj: %dx rnd img w/ dim(%d,%d)\n", aln_cfg.sbj_num, aln_cfg.img_dim, aln_cfg.img_dim );
    float** sbj_data = create_rnd_data( aln_cfg.sbj_num, aln_cfg.img_dim );
    //print_data( (const float**)sbj_data, aln_cfg.sbj_num, aln_cfg.img_dim );
    pre_align_fetch( (const float**)sbj_data, aln_cfg.sbj_num, "sbj_batch" );

    // ref data
    printf( "create ref: %dx rnd img data w/ dim(%d,%d)\n", aln_cfg.ref_num, aln_cfg.img_dim, aln_cfg.img_dim );
    float** ref_data = create_rnd_data( aln_cfg.ref_num, aln_cfg.img_dim );
    //print_data( (const float**)ref_data, aln_cfg.ref_num, aln_cfg.img_dim );
    pre_align_fetch( (const float**)ref_data, aln_cfg.ref_num, "ref_batch" );

    // time alignment call
    printf( "run alignment\n" );

    float time;
    cudaEvent_t start, stahp;
    CUDA_ERR_CHK( cudaEventCreate(&start) );
    CUDA_ERR_CHK( cudaEventCreate(&stahp) );
    CUDA_ERR_CHK( cudaEventRecord(start, 0) );

    pre_align_run( 0, aln_cfg.sbj_num );

    CUDA_ERR_CHK( cudaEventRecord(stahp, 0) );
    CUDA_ERR_CHK( cudaEventSynchronize(stahp) );
    CUDA_ERR_CHK( cudaEventElapsedTime(&time, start, stahp) );
    printf( "aln time: %.3fs\n", (time/1000.0) );

    // clean up
    gpu_clear();

    delete_rnd_data( sbj_data, aln_cfg.sbj_num, aln_cfg.img_dim );
    delete_rnd_data( ref_data, aln_cfg.ref_num, aln_cfg.img_dim );

    // exit
    printf( "all done\n" );
    return EXIT_SUCCESS;
}

// original alignment
int main_0( int argc, const char** argv ){

    CUDA_ERR_CHK( cudaSetDevice(CUDA_DEVICE_ID) );
    print_gpu_info(CUDA_DEVICE_ID);

    // param
    AlignConfig data_cfg = AlignConfig();
    data_cfg.sbj_num = 10000;
    data_cfg.ref_num = 1;
    data_cfg.shift_rng_x = 2;
    data_cfg.shift_rng_y = 2;

    // data
    ImageStack sbj_stack( data_cfg.sbj_num, data_cfg.img_dim );
    ImageStack ref_stack( data_cfg.ref_num, data_cfg.img_dim );

    const float** sbj_ptr_list = sbj_stack.img_ptr_list();
    const float** ref_ptr_list = ref_stack.img_ptr_list();

    // memcheck
    assert( ref_free_alignment_2D_size_check(&data_cfg, CUDA_DEVICE_ID, 0.9f, true) );

    // prepare output array and fill w/ class id info for the alignment
    unsigned int img_per_class = (data_cfg.sbj_num%data_cfg.ref_num == 0)
        ? data_cfg.sbj_num/data_cfg.ref_num
        : data_cfg.sbj_num/data_cfg.ref_num+1;

    int sbj_cid_list[ data_cfg.sbj_num ];
    for( unsigned int i=0; i<data_cfg.sbj_num; i++ )
        sbj_cid_list[i] = i/img_per_class;

    // initialize timing
    float time;
    cudaEvent_t start, stahp;
    CUDA_ERR_CHK( cudaEventCreate(&start) );
    CUDA_ERR_CHK( cudaEventCreate(&stahp) );
    CUDA_ERR_CHK( cudaEventRecord(start, 0) );

    // initialize alignment
    AlignParam* aln_param = ref_free_alignment_2D_init( &data_cfg, sbj_ptr_list, ref_ptr_list, sbj_cid_list, 0 );

    // iterate
    unsigned int iterations=10;
    for( unsigned int i=0; i<iterations; i++ ){
        printf( "\r%d..", (i+1) );
        fflush( stdout );
        ref_free_alignment_2D_filter_references( 0.1f, 0.2f );
        ref_free_alignment_2D();
    }
    printf("\n");

    // free resources
    gpu_clear();

    // report time
    CUDA_ERR_CHK( cudaEventRecord(stahp, 0) );
    CUDA_ERR_CHK( cudaEventSynchronize(stahp) );
    CUDA_ERR_CHK( cudaEventElapsedTime(&time, start, stahp) );
    printf( "GPU time: %.3fs (%.3fs per total iteration; %.3fs per class; %.3fs per class iteration; ~%.3fs for 30 iterations)\n", 
        (time/1000.0),
        (time/1000.0)/iterations,
        (time/1000.0)/data_cfg.ref_num,
        (time/1000.0)/(iterations*data_cfg.ref_num),
        30.0*(time/1000.0)/iterations );

    printf( "All done.\n" );
    return EXIT_SUCCESS;
}

