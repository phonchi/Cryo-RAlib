
/******************************************************************************

GPU based multireference alignment.

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

#ifndef GPU_ALN_COMMON
#define GPU_ALN_COMMON

//===================================================================[ import ]

// C core
#include <cstdio>
#include <cstdlib>
#include <cassert>

// C++ core
#include <array>
#include <vector>
#include <iostream>

// CUDA
#include <cufft.h>
#include <cublas_v2.h>

//===============================================================[ namespaces ]

using namespace std;

//===========================================================[ default values ]

#define DEFAULT_IMG_DIM       76   // particle images are downsized to dimensions (DEFAULT_IMG_DIM, DEFAULT_IMG_DIM) [formerly NX,NY]
#define DEFAULT_RING_NUM      32   // number of rings used when sampling to polar [formerly NRING]
#define DEFAULT_RING_LEN     256   // number of sampling points aling each ring when sampling to polar [formerly RING_LENGTH]

#define DEFAULT_SHIFT_STEP     1   // step size when applying shifts to (subject) images [formerly STEP]
#define DEFAULT_SHIFT_RNG_X    1   // shift range in x direction (images are shifted from -range to +range)
#define DEFAULT_SHIFT_RNG_Y    1   // shift range in y direction

#define PI 3.141592653589793f
#define DEG2RAD PI/180.0f
#define RAD2DEG 180.0f/PI

//==========================================================[ utility structs ]

struct AlignConfig{
    // data info
    unsigned int sbj_num;   // number of subject images
    unsigned int ref_num;   // number of reference images
    unsigned int img_dim = DEFAULT_IMG_DIM;
    // polar sampling parameters
    unsigned int ring_num = DEFAULT_RING_NUM;
    unsigned int ring_len = DEFAULT_RING_LEN;
    // shift parameters
    float shift_step  = DEFAULT_SHIFT_STEP;
    float shift_rng_x = DEFAULT_SHIFT_RNG_X;
    float shift_rng_y = DEFAULT_SHIFT_RNG_Y;
};

struct AlignParam{
    int   sbj_id=-1;
    int   ref_id=-1;
    float shift_x=0;
    float shift_y=0;
    float angle=0.0f;
    bool  mirror=false;
};

//========================================================[ utility functions ]

ostream& operator<<(ostream& os, const AlignParam& aln_param);

float* generate_polar_sampling_points( const unsigned int ring_num, const unsigned int ring_len );

vector<array<float,2>>* generate_shift_array(
    const float shift_rng_x,
    const float shift_rng_y,
    const float shift_step );

//--------------------------------------------------------[ cuda error checks ]

// CUDA error checking for library functions
#define CUDA_ERR_CHK(func){ cuda_assert( (func), __FILE__, __LINE__ ); }
void cuda_assert( const cudaError_t cuda_err, const char* file, const int line );

// CUDA generic error checking (used after kernel calls)
#define KERNEL_ERR_CHK(){ kernel_assert( __FILE__, __LINE__ ); }
void kernel_assert( const char* file, const int line );

// CUDA error checking for CUBLAS library calls
#define CUBLAS_ERR_CHK(func){ cublas_assert( (func), __FILE__, __LINE__ ); }
void cublas_assert( const cublasStatus_t cublas_err, const char* file, const int line );

// CUDA error checking for CUFFT library calls
#define CUFFT_ERR_CHK(func){ cufft_assert( (func), __FILE__, __LINE__ ); }
void cufft_assert( const cufftResult_t cufft_err, const char* file, const int line );

//-----------------------------------------------------------[ cuda utilities ]

// simple function to initialize CUDA device
void initialize_device();

// print specifications of used CUDA device
extern "C" void print_gpu_info( const unsigned int device_idx );

// print available memory on the device
void print_device_memcheck();

// get default memory pitch of the selected device
size_t get_device_mem_pitch( const unsigned int device_id=0 );


//=========================================================[ ImageStack class ]

class ImageStack{

    protected:
        // raw data
        unsigned int img_num;
        float*       img_data;
        unsigned int img_dim_x;
        unsigned int img_dim_y;

    public:
        ImageStack( const unsigned int img_num, unsigned int const img_dim );
        ImageStack();
       ~ImageStack();

        float*  img_ptr( const unsigned int img_idx ) const;
        const float** img_ptr_list() const;
};


//===================================================[ cuda stream pool class ]

class CudaStreamPool{

    private:
        unsigned int  stream_num;
        cudaStream_t* cuda_streams;

    public:
        CudaStreamPool( const unsigned int device_idx, const unsigned int stream_num );
       ~CudaStreamPool();

        unsigned int size() const;
        cudaStream_t get_stream( const unsigned int idx ) const;
        void synchronize() const;
};

//====================================================================[ other ]

__device__ void cu_max_idx_op(
    const unsigned int i,
    const unsigned int off,
    float*             shared_data,
    unsigned int*      shared_idx );

__global__ void cu_max_idx_batch(
    const float*  data,
    unsigned int  len,
    unsigned int* idx );

#endif
