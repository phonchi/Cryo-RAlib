
/******************************************************************************

GPU based multireference alignment

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

#include "gpu_aln_common.h"


//========================================================[ utility functions ]

// output stream operator for the alignment parameter struct
ostream& operator<<( ostream& os, const AlignParam& aln_param ){
    os << "s_" << aln_param.sbj_id << "/r_" << aln_param.ref_id << "::(" << aln_param.shift_x << "," << aln_param.shift_y << "/" << aln_param.angle << ")";
    if( aln_param.mirror )
        os << "[M]";
    return os;
}

float* generate_polar_sampling_points( const unsigned int ring_num, const unsigned int ring_len ){
    /* Generate an array of coordinates used for resampling images into polar
    cooordinates. The memory layout is:

        [--<ring_len*2 >--][--<ring_len*2 >--] .. [--<ring_len*2 >--]

    Where each block holds <ring_len> number of float pairs and the first block
    holding the coordinates of the innermost ring. */

    // allocate memory in unified memory space
    unsigned int no_of_sample_pts = ring_len*ring_num*2;
    float* u_polar_sample_coords=NULL;
    CUDA_ERR_CHK( cudaMallocManaged(&u_polar_sample_coords, no_of_sample_pts*sizeof(float)) );
    // create the polar sampling points
    for( unsigned int i=0; i<ring_num; i++ ){
        for( unsigned int j=0; j<ring_len; j++){
            unsigned int idx = i*ring_len+j;
            float angle = float(j)/float(ring_len)*3.14159265358979*2;
            u_polar_sample_coords[idx*2  ] = cosf(angle) *(i+1)*float(ring_num)/float(ring_num);
            u_polar_sample_coords[idx*2+1] = sinf(angle) *(i+1)*float(ring_num)/float(ring_num);            
        }
    }
    return u_polar_sample_coords;
}

vector<array<float,2>>* generate_shift_array(
    const float shift_rng_x,
    const float shift_rng_y,
    const float shift_step)
{
    /* Returns an array of shift pairs given x and y ranges. Each value depicts
    the (included) +/- limits of the shift range, i.e., we shift within the
    ranges [-shift_range_x, shift_range_x] and [-shift_range_y, shift_range_y].

    NOTE: The shift array resides in host memory as it is never accessed on the
    device. Instead, kernels are passed the relevant shift values as arguments.
    */
    vector<array<float,2>>* shifts = new vector<array<float,2>>;
    for( float s_x = -shift_rng_x; s_x <= shift_rng_x; s_x+=shift_step ){
        for( float s_y = -shift_rng_y; s_y <= shift_rng_y; s_y+=shift_step ){
            array<float,2> s = { s_x, s_y };
            shifts->push_back( s );
        }
    }
    return shifts;
}

//--------------------------------------------------------[ cuda error checks ]

// error checking for cuda library functions
void cuda_assert( const cudaError_t cu_err, const char* file, int line ){
    if( cu_err != cudaSuccess ){
        fprintf( stderr, "\nGPU ERROR! \'%s\' (err code %d) in file %s, line %d.\n\n", cudaGetErrorString(cu_err), cu_err, file, line );
        exit( EXIT_FAILURE );
    }
}

// error checking for cuda kernel executions
void kernel_assert( const char* file, const int line ){
    cudaError cu_err = cudaGetLastError();
    if( cu_err != cudaSuccess ){
        fprintf( stderr, "\nGPU KERNEL ERROR! \'%s\' (err code %d) in file %s, line %d.\n\n", cudaGetErrorString(cu_err), cu_err, file, line );
        exit(EXIT_FAILURE);
    }
}

// error checking for CUBLAS library calls
void cublas_assert( const cublasStatus_t cublas_err, const char* file, const int line ){
    if( cublas_err != CUBLAS_STATUS_SUCCESS ){
        fprintf( stderr, "\nCUBLAS EXECUTION ERROR in file %s, line %d:\n", file, line );
        switch( cublas_err ){
            case CUBLAS_STATUS_NOT_INITIALIZED:  fprintf( stderr, "CUBLAS error: CUBLAS_STATUS_NOT_INITIALIZED\n"  ); break;
            case CUBLAS_STATUS_ALLOC_FAILED:     fprintf( stderr, "CUBLAS error: CUBLAS_STATUS_ALLOC_FAILED\n"     ); break;
            case CUBLAS_STATUS_INVALID_VALUE:    fprintf( stderr, "CUBLAS error: CUBLAS_STATUS_INVALID_VALUE\n"    ); break;
            case CUBLAS_STATUS_ARCH_MISMATCH:    fprintf( stderr, "CUBLAS error: CUBLAS_STATUS_ARCH_MISMATCH\n"    ); break;
            case CUBLAS_STATUS_MAPPING_ERROR:    fprintf( stderr, "CUBLAS error: CUBLAS_STATUS_MAPPING_ERROR\n"    ); break;
            case CUBLAS_STATUS_EXECUTION_FAILED: fprintf( stderr, "CUBLAS error: CUBLAS_STATUS_EXECUTION_FAILED\n" ); break;
            case CUBLAS_STATUS_INTERNAL_ERROR:   fprintf( stderr, "CUBLAS error: CUBLAS_STATUS_INTERNAL_ERROR\n"   ); break;
            case CUBLAS_STATUS_NOT_SUPPORTED:    fprintf( stderr, "CUBLAS error: CUBLAS_STATUS_NOT_SUPPORTED\n"    ); break;
            case CUBLAS_STATUS_LICENSE_ERROR:    fprintf( stderr, "CUBLAS error: CUBLAS_STATUS_LICENSE_ERROR\n"    ); break;
        }
        exit( EXIT_FAILURE );
    }
}

// error checking for CUFFT library calls
void cufft_assert( const cufftResult_t cufft_err, const char* file, const int line ){
    if( cufft_err != CUFFT_SUCCESS ){
        fprintf( stderr, "\nCUFFT EXECUTION ERROR in file %s, line %d:\n", file, line );
        switch( cufft_err ){
            case CUFFT_INVALID_PLAN:    fprintf( stderr, "CUFFT error (CUFFT_INVALID_PLAN): CUFFT was passed an invalid plan handle.\n" ); break;
            case CUFFT_ALLOC_FAILED:    fprintf( stderr, "CUFFT error (CUFFT_ALLOC_FAILED): CUFFT failed to allocate GPU or CPU memory.\n" ); break;
            case CUFFT_INVALID_TYPE:    fprintf( stderr, "CUFFT error (CUFFT_INVALID_TYPE): No longer used.\n" ); break;
            case CUFFT_INVALID_VALUE:   fprintf( stderr, "CUFFT error (CUFFT_INVALID_VALUE): User specified an invalid pointer or parameter.\n" ); break;
            case CUFFT_INTERNAL_ERROR:  fprintf( stderr, "CUFFT error (CUFFT_INTERNAL_ERROR): Driver or internal CUFFT library error.\n" ); break;
            case CUFFT_EXEC_FAILED:     fprintf( stderr, "CUFFT error (CUFFT_EXEC_FAILED): Failed to execute an FFT on the GPU.\n" ); break;
            case CUFFT_SETUP_FAILED:    fprintf( stderr, "CUFFT error (CUFFT_SETUP_FAILED): The CUFFT library failed to initialize.\n" ); break;
            case CUFFT_INVALID_SIZE:    fprintf( stderr, "CUFFT error (CUFFT_INVALID_SIZE): User specified an invalid transform size.\n" ); break;
            case CUFFT_UNALIGNED_DATA:  fprintf( stderr, "CUFFT error (CUFFT_UNALIGNED_DATA): No longer used.\n" ); break;
            case CUFFT_INVALID_DEVICE:  fprintf( stderr, "CUFFT error (CUFFT_INVALID_DEVICE): Execution of a plan was on different GPU than plan creation.\n" ); break;
            case CUFFT_PARSE_ERROR:     fprintf( stderr, "CUFFT error (CUFFT_PARSE_ERROR): Internal plan database error .\n" ); break;
            case CUFFT_NO_WORKSPACE:    fprintf( stderr, "CUFFT error (CUFFT_NO_WORKSPACE): Workspace has been provided prior to plan execution.\n" ); break;
            case CUFFT_NOT_IMPLEMENTED: fprintf( stderr, "CUFFT error (CUFFT_NOT_IMPLEMENTED): Function does not implement functionality for parameters given..\n" ); break;
            case CUFFT_LICENSE_ERROR:   fprintf( stderr, "CUFFT error (CUFFT_LICENSE_ERROR): Used in previous versions.\n" ); break;
            case CUFFT_NOT_SUPPORTED:   fprintf( stderr, "CUFFT error (CUFFT_NOT_SUPPORTED): is not supported for parameters given.\n" ); break;
            case CUFFT_INCOMPLETE_PARAMETER_LIST:  fprintf( stderr, "CUFFT error (CUFFT_INCOMPLETE_PARAMETER_LIST): Missing parameters in call.\n" ); break;
        }
    }
}

//-----------------------------------------------------------[ cuda utilities ]

// simple hello-world function to initialize CUDA
void initialize_device(){
    float  f=1.0;
    float *dev_ptr=NULL;
    CUDA_ERR_CHK( cudaMalloc(&dev_ptr, 100*1024*1024) );
    CUDA_ERR_CHK( cudaMemcpy(&dev_ptr[23], &f, sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_ERR_CHK( cudaDeviceSynchronize() );
    float g = 0.0;
    CUDA_ERR_CHK( cudaMemcpy(&g, &dev_ptr[23], sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_ERR_CHK( cudaFree(dev_ptr) );
    printf( "CUDA initialized (%.1f|1.0).\n", g );
}

// print specs of selected CUDA device
extern "C" void print_gpu_info( const unsigned int device_idx ){
    int driver, runtime;
    CUDA_ERR_CHK( cudaDriverGetVersion(&driver) );
    CUDA_ERR_CHK( cudaRuntimeGetVersion(&runtime) );
    printf( "CUDA driver  version: %d.%d\n", driver/1000, driver%100/10 );
    printf( "CUDA runtime version: %d.%d\n", runtime/1000, runtime%100/10 );
    cudaDeviceProp dev_prop;
    CUDA_ERR_CHK( cudaGetDeviceProperties(&dev_prop, device_idx) );
    printf( "CUDA device[%d] info for %s:\n", device_idx, dev_prop.name );
    printf( "Compute capability: %d.%d\n", dev_prop.major, dev_prop.minor );
    printf( "No. of multiprocessors: %d\n", dev_prop.multiProcessorCount );
    printf( "____[ Flags ]____\n" );
    printf( "Mapping of host memory into CUDA address space: %d (cudaHostmalloc)\n", dev_prop.canMapHostMemory );
    printf( "Multiple kernel execution supported: %d (NOT GUARANTEED)\n", dev_prop.concurrentKernels );
    printf( "Run time limit for kernel execution: %d\n", dev_prop.kernelExecTimeoutEnabled );
    printf( "Unified addressing available: %d\n", dev_prop.unifiedAddressing );
    printf( "Managed memory available: %d\n", dev_prop.managedMemory );
    printf( "____[ Memory ]____\n" );
    printf( "Total global memory: %.3f GB\n", float(dev_prop.totalGlobalMem)/(1024*1024*1024) );
    printf( "Total constant memory: %.0f KB\n", float(dev_prop.totalConstMem/1024) );
    printf( "Shared memory per block: %zu bytes (%zu KB)\n", dev_prop.sharedMemPerBlock, dev_prop.sharedMemPerBlock/1024 );
    printf( "Shared memory per multiprocessor: %zu bytes (%zu KB)\n", dev_prop.sharedMemPerMultiprocessor, dev_prop.sharedMemPerMultiprocessor/1024 );
    printf( "____[ Grid ]____\n" );
    printf( "Maximum grid dim:   (%d, %d, %d)\n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2] );
    printf( "Maximum thread dim: (%d, %d, %d)\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2] );
    printf( "Max. no. of threads per block: %d\n", dev_prop.maxThreadsPerBlock );
    printf( "Max. no. of threads per multiprocessor: %d\n", dev_prop.maxThreadsPerMultiProcessor );
    printf( "Max. no. of registers available per block: %d (shared by all blocks resident on a multiprocessor)\n", dev_prop.regsPerBlock );
    printf( "Warp size: %d\n", dev_prop.warpSize );
}

// print available and total memory
void print_device_memcheck(){
    // make sure the memcheck is up to date
    CUDA_ERR_CHK( cudaDeviceSynchronize() );
    fflush(stdout); fflush(stderr);
    // now the actual device query
    size_t mem_avl=0, mem_ttl=0;
    CUDA_ERR_CHK( cudaMemGetInfo(&mem_avl, &mem_ttl) );
    fprintf( stderr, "CUDA memcheck: %.2fmb/%.2fmb available.\n", float(mem_avl)/(1024.0f*1024.0f), float(mem_ttl)/(1024.0f*1024.0f) );
}

// get default memory pitch of the selected device
size_t get_device_mem_pitch( const unsigned int device_id ){
    float* ptr;
    size_t pitch;
    CUDA_ERR_CHK( cudaSetDevice(device_id) );
    CUDA_ERR_CHK( cudaMallocPitch(&ptr, &pitch, sizeof(float), 1) );
    CUDA_ERR_CHK( cudaFree(ptr) );
    return pitch;
}


//=========================================================[ ImageStack class ]

ImageStack::ImageStack( const unsigned int num, const unsigned int img_dim ){
    /*
    Constructor to create a stack containing <img_num> images filled with
    random data. Used for timing & debugging purposes.
    */

    fprintf( stderr, "Creating %d random images of dim(%d,%d).\n", num, img_dim, img_dim );
    
    img_num   = num;
    img_dim_x = img_dim_y = img_dim;
    img_data  = (float*)malloc( img_num*img_dim_x*img_dim_y*sizeof(float) );
    
    for( unsigned int i=0; i<img_num; i++ ){
        for( unsigned int row=0; row<img_dim_y; row++ ){
            for( unsigned int col=0; col<img_dim_x; col++ )
                img_data[ i*img_dim_x*img_dim_y + row*img_dim_x + col ] = rand()%255;
        }
    }
}

ImageStack::ImageStack(){
    img_num = img_dim_x = img_dim_y = 0;
    img_data = NULL;
}

ImageStack::~ImageStack(){
    if( img_data != NULL ){
        free(img_data);
        img_data = NULL;
    }
}

//---------------------------------------------------------[ public interface ]

float* ImageStack::img_ptr( const unsigned int img_idx ) const{
    /*
    Returns a pointer to img_i within the continuous ImageStack data array.
    */
    assert( img_data != NULL );
    assert( img_idx < img_num );
    return &img_data[ img_idx*img_dim_x*img_dim_y ];
}

const float** ImageStack::img_ptr_list() const{
    const float** ptr_list = (const float**)malloc( img_num*sizeof(float*) );
    for( unsigned int i=0; i<img_num; i++ )
        ptr_list[i] = img_ptr(i);
    return ptr_list;
}


//===================================================[ cuda stream pool class ]

CudaStreamPool::CudaStreamPool( const unsigned int device_idx, const unsigned int stream_num ){
    /* This class allows the modularized creation, synchronization, and 
    destruction of a set of CUDA streams. The primariy idea of this is to have
    this class hold streams that can be re-used in different parts of the code.

    Args:

        device_idx: Index of the CUDA device the streams are being created on.
           NOTE: A stream created on a specific device N will not be usable on
           another device!

        stream_num: Number of streams in the pool. If no number is specified
           the pool will contain as many numbers as the selected device has 
           microprocessors.

    NOTE: Class functions will always assume that the CUDA device the streams
    have been created on is selected when they are called. (This is only an 
    issue when using multiple GPUs which, right now, the C code below is not 
    doing.) */

    // number of streams in the pool
    if( stream_num == 0 ){
        cudaDeviceProp dev_prop;
        CUDA_ERR_CHK( cudaGetDeviceProperties(&dev_prop, device_idx) );
        this->stream_num = dev_prop.multiProcessorCount;
    }
    else
        this->stream_num = stream_num;

    // create streams
    cuda_streams = (cudaStream_t*)malloc( stream_num * sizeof(cudaStream_t) );
    for( unsigned int i=0; i< stream_num; i++ )
        CUDA_ERR_CHK( cudaStreamCreate(&cuda_streams[i]) );
}

CudaStreamPool::~CudaStreamPool(){
    if( cuda_streams != NULL ){
        for( unsigned int i=0; i<stream_num; i++ )
            CUDA_ERR_CHK( cudaStreamDestroy(cuda_streams[i]) );
        free( cuda_streams );
        cuda_streams = NULL;
    }
}

unsigned int CudaStreamPool::size() const {
    return stream_num;
}

cudaStream_t CudaStreamPool::get_stream( const unsigned int idx ) const {
    assert( idx < stream_num );
    return cuda_streams[idx];
}

void CudaStreamPool::synchronize() const {
    for( unsigned int i=0; i<stream_num; i++ )
        CUDA_ERR_CHK( cudaStreamSynchronize(cuda_streams[i]) );
}


//====================================================================[ other ]

/*
The following is an efficient segmented reduction to find idx of max values 
given a 2D-layout array. However, since we would apply this function to arrays
of substantial size, it is acutally faster to simply go with cublasIsamax()
calls instead.

   This means both cu_max_idx_op() and cu_max_idx_batch() are not used and
could be removed. I'm keeping them here, however, in case we need them, or
something similar, in the future.
*/

// max operation: find max val and max idx between two the values at [i] and [i+off]
__device__ void cu_max_idx_op( const unsigned int i, const unsigned int off, float* shared_data, unsigned int* shared_idx ){
    /*   DATA SEGMENT          INDEX SEGMENT
        |=x========:=x========|-x--------:-x--------|    - shared memory array has two parts (':' marks halfway within each segment)
          |__________|          |__________|             - value of <off> is depicted as range of |__________|
          |                     |                        - index half starts at <i+offx2>
          |          ___________|                        - No. of threads == value of <off>
          v          v                                   - NOTE: shared_data and shared_idx are the same address!
        |=x========|-x--------|
          ^          ^          ^          ^
         [i]        [i+off]    [i+off*2]  [i+off*3]
    */
    shared_data[i]     = (shared_data[i] > shared_data[i+off]) ? shared_data[i]       : shared_data[i+off];    // val update
    shared_idx [i+off] = (shared_data[i] > shared_data[i+off]) ? shared_idx [i+off*2] : shared_idx [i+off*3];  // idx update
}

// batch-wise max_idx reduction: each block finds the max val and its idx within a range of size <len> within a continuous data array
__global__ void cu_max_idx_batch( const float* data, unsigned int len, unsigned int* idx ){
    /*
    Based on: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    */
    // shared memory
    extern __shared__ float        shared_data[];
    extern __shared__ unsigned int shared_idx [];
    // grid id and data pointer
    unsigned int tid = threadIdx.x;
    unsigned int data_idx = blockIdx.x * len + tid;
    // load data into shared memory (first half stores values, second half stores value indices in the original array)
    unsigned int off = blockDim.x;
    shared_data[tid]     = (data[data_idx] > data[data_idx+off]) ? data[data_idx] : data[data_idx+off];
    shared_idx [tid+off] = (data[data_idx] > data[data_idx+off]) ? tid : tid+off;
    __syncthreads();
    // unrolled reduction (syncthreads only needed above warp size)
    if(tid < 64) cu_max_idx_op(tid, 64, shared_data, shared_idx); __syncthreads();
    if(tid < 32) cu_max_idx_op(tid, 32, shared_data, shared_idx);
    if(tid < 16) cu_max_idx_op(tid, 16, shared_data, shared_idx);
    if(tid <  8) cu_max_idx_op(tid,  8, shared_data, shared_idx);
    if(tid <  4) cu_max_idx_op(tid,  4, shared_data, shared_idx);
    if(tid <  2) cu_max_idx_op(tid,  2, shared_data, shared_idx);
    if(tid <  1) cu_max_idx_op(tid,  1, shared_data, shared_idx);
    // shared_idx[0] holds the max value; shared_idx[1] holds the index of the max in the original array
    if(tid == 0) idx[blockIdx.x] = shared_idx[1];
}
