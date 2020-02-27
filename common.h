#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "tests.cc"
#include <assert.h>


/* 
##################################
Three kinds of compute Cmode
 *  cpu
 *  gpu
 *  fpga
*/
enum struct CCmode{
  gpu,
  cpu,
  fpga
};

/* 
##################################
Record time
*/

struct Timer{
  double t1;
  Timer(): t1(omp_get_wtime()) {}
  double stop(){return omp_get_wtime() - t1;}
};

/* 
##################################
input_instance
* x0: scalar 
* x1: scalar
* num: Big Number
*/
template<uint32_t bits>
typedef struct input_instance{
    public：
      cgbn_mem_t<bits> *x0;
      cgbn_mem_t<bits> *x1;
      cgbn_mem_t<bits> num;
      Cmode type;
      input_instance(int count, Cmode type){
        this->type = type;
        if (this->type == Cmode::cpu){
          this->x0 = (cgbn_mem_t<bits> *)malloc(sizeof(cgbn_mem_t<bits>)*count);
          this->x1 = (cgbn_mem_t<bits> *)malloc(sizeof(cgbn_mem_t<bits>)*count);
        }else if (this->type == Cmode::gpu){
          CUDA_CHECK(cudaMalloc((void **)&this->x0, sizeof(cgbn_mem_t<bits>)*count));
          CUDA_CHECK(cudaMalloc((void **)&this->x1, sizeof(cgbn_mem_t<bits>)*count));
        }else if (this->type == Cmode::fpga){
          //fpga
        }else{
          printf("Error: Unknown Cmode.");
          exit(0);
        }
      }
      ~input_instance(){
        if (this->type == Cmode::cpu){
          free(x0);
          free(x1);
        }else if (this->type == Cmode::gpu){
          CUDA_CHECK(cudaFree(x0));
          CUDA_CHECK(cudaFree(x1));
        }else{
          //fpga
        }
      }
  };

/* 
##################################
mem_results
* r: scalar 
*/
template<uint32_t bits>
typedef struct mem_results{
    public:
      cgbn_mem_t<bits> *r;
      Cmode type;
      mem_results(int count, Cmode type){
        this->type = type;
        if (this->type == Cmode::cpu){
          this->r = (cgbn_mem_t<bits> *)malloc(sizeof(cgbn_mem_t<bits>)*count);
        }else if (this->type == Cmode::gpu){
          CUDA_CHECK(cudaMalloc((void **)&this->r, sizeof(cgbn_mem_t<bits>)*count));
        }else if (this->type == Cmode::fpga){
          //fpga
        }else{
          printf("Error: Unkown Cmode.");
          exit(0);
        }
      }
      ~mem_results(){
        if (this->type == Cmode::cpu){
          free(r);
        }else if (this->type == Cmode::gpu){
          CUDA_CHECK(cudaFree(r));
        }else{
          //fpga
        }
      }
  };

/* 
##################################
supported_size
* size: size of each instance
*/

bool supported_size(uint32_t size) {
  return size==128 || size==256 || size==512 || 
         size==1024 || size==2048 || size==3072 || size==4096 ||
         size==5120 || size==6144 || size==7168 || size==8192;
}

/* 
##################################
supported_tpi_size
* tpi: threads per instance
* size: size of each instance
*/

bool supported_tpi_size(uint32_t tpi, uint32_t size) {
  if(size==128 && tpi==4)
    return true;
  else if(size==256 && (tpi==4 || tpi==8))
    return true;
  else if(size==512 && (tpi==4 || tpi==8 || tpi==16))
    return true;
  else if(size==1024 && (tpi==8 || tpi==16 || tpi==32))
    return true;
  else if(size==2048 && (tpi==8 || tpi==16 || tpi==32))
    return true;
  else if(size==3072 && (tpi==16 || tpi==32))
    return true;
  else if(size==4096 && (tpi==16 || tpi==32))
    return true;
  else if(size==5120 && tpi==32)
    return true;
  else if(size==6144 && tpi==32)
    return true;
  else if(size==7168 && tpi==32)
    return true;
  else if(size==8192 && tpi==32)
    return true;
  return false;
}