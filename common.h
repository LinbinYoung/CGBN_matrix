#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include <assert.h>
#include "gpu_support.h"

/* 
##################################
Different Compute Type
*/
typedef enum {
  test_all,
  test_unknown,
  xt_add,
  xt_addui,
  xt_mul,
  xt_sub,
  xt_accumulate,
  xt_div_qr,
  xt_sqrt,
  xt_powm_odd,
  xt_mont_reduce,
  xt_gcd,
  xt_modinv,
} Compute_Type;

#define XT_FIRST xt_add
#define XT_LAST  xt_modinv
#define XT_CURRENT xt_mul

/* 
##################################
parse compute mode
*/
Compute_Type parse_compute_type(const char *name) {
  if(strcmp(name, "add")==0)
    return xt_add;
  else if (strcmp(name, "addui"))
    return xt_addui;
  else if(strcmp(name, "sub")==0)
    return xt_sub;
  else if(strcmp(name, "accumulate")==0)
    return xt_accumulate;
  else if(strcmp(name, "mul")==0)
    return xt_mul;
  else if(strcmp(name, "div_qr")==0)
    return xt_div_qr;
  else if(strcmp(name, "sqrt")==0)
    return xt_sqrt;
  else if(strcmp(name, "powm_odd")==0)
    return xt_powm_odd;
  else if(strcmp(name, "mont_reduce")==0)
    return xt_mont_reduce;
  else if(strcmp(name, "gcd")==0)
    return xt_gcd;
  else if(strcmp(name, "modinv")==0)
    return xt_modinv;
  return test_unknown;
}

/* 
##################################
return compute mode
*/
const char *actual_compute_name(Compute_Type test) {
  switch(test) {
    case xt_add:
      return "add";
    case xt_addui:
      return "addui";
    case xt_sub:
      return "sub";
    case xt_accumulate:
      return "accumulate";
    case xt_mul:
      return "mul";
    case xt_div_qr:
      return "div_qr";
    case xt_sqrt:
      return "sqrt";
    case xt_powm_odd:
      return "powm_odd";
    case xt_mont_reduce:
      return "mont_reduce";
    case xt_gcd:
      return "gcd";
    case xt_modinv:
      return "modinv";
  }
  return "unknown";
}

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
###################################
Data Base
* x0: scalar 
* x1: scalar
* num: Big Number
*/
template<uint32_t bits>
class DataBase{
  public:
    cgbn_mem_t<bits> *x0;
    cgbn_mem_t<bits> *x1;
    cgbn_mem_t<bits> num;
    DataBase(uint32_t count){}
    virtual ~DataBase(){}
};

/* 
##################################
GPU_Data
* x0: scalar 
* x1: scalar
* num: Big Number
*/
template<uint32_t bits>
class GPU_Data : public DataBase<bits>{
  public:
    GPU_Data(int count):DataBase<bits>(count){
      CUDA_CHECK(cudaMalloc((void **)&this->x0, sizeof(cgbn_mem_t<bits>)*count));
      CUDA_CHECK(cudaMalloc((void **)&this->x1, sizeof(cgbn_mem_t<bits>)*count));
    }
    ~GPU_Data(){
      CUDA_CHECK(cudaFree(this->x0));
      CUDA_CHECK(cudaFree(this->x1));
    }
};

/* 
##################################
CPU_data
* x0: scalar 
* x1: scalar
* num: Big Number
*/
template<uint32_t bits>
class CPU_Data : public DataBase<bits>{
  public:
    CPU_Data(int count):DataBase<bits>(count){
      this->x0 = (cgbn_mem_t<bits> *)malloc(sizeof(cgbn_mem_t<bits>)*count);
      this->x1 = (cgbn_mem_t<bits> *)malloc(sizeof(cgbn_mem_t<bits>)*count);
    }
    ~CPU_Data(){
      free(this->x0);
      free(this->x1);
    }
};

/* 
##################################
FPGA_Data
* x0: scalar 
* x1: scalar
* num: Big Number
*/
template<uint32_t bits>
class FPGA_Data : public DataBase<bits>{
    public:
    FPGA_Data(int count):DataBase<bits>(count){
      this->x0 = (cgbn_mem_t<bits> *)malloc(sizeof(cgbn_mem_t<bits>)*count);
      this->x1 = (cgbn_mem_t<bits> *)malloc(sizeof(cgbn_mem_t<bits>)*count);
    }
    ~FPGA_Data(){
      free(this->x0);
      free(this->x1);
    }
};

/*
###################################
Result Base
* r: scalar 
*/
template<uint32_t bits>
class ResultBase{
  public:
    cgbn_mem_t<bits> *r;
    ResultBase(uint32_t count){}
    virtual ~ResultBase(){}
};

/* 
##################################
GPU_result
* r: scalar 
*/
template<uint32_t bits>
class GPU_result : public ResultBase<bits>{
  public:
    GPU_result(int count):ResultBase<bits>(count){
      CUDA_CHECK(cudaMalloc((void **)&this->r, sizeof(cgbn_mem_t<bits>)*count));
    }
    ~GPU_result(){
      CUDA_CHECK(cudaFree(this->r));
    }
};

/* 
##################################
CPU_result
* r: scalar 
*/
template<uint32_t bits>
class CPU_result : public ResultBase<bits>{
  public:
    CPU_result(int count):ResultBase<bits>(count){
      this->r = (cgbn_mem_t<bits> *)malloc(sizeof(cgbn_mem_t<bits>)*count);
    }
    ~CPU_result(){
      free(this->r);
    }
};


/* 
##################################
FPGA_result
* r: scalar 
*/
template<uint32_t bits>
class FPGA_result : public ResultBase<bits>{
  public:
    FPGA_result(int count):ResultBase<bits>(count){
      this->r = (cgbn_mem_t<bits> *)malloc(sizeof(cgbn_mem_t<bits>)*count);
    }
    ~FPGA_result(){
      free(this->r);
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

/*
 * Function: from_mpz
 * Description: load value from mpz object into cgbn_mem_t
 * Para: 
 *   words: target room
 *   count: number of words transferred
 *   value: mpz value
 */ 
void from_mpz(uint32_t *words, uint32_t count, mpz_t value) {
  size_t written;
  if(mpz_sizeinbase(value, 2)>count*32) {
    fprintf(stderr, "from_mpz failed -- result does not fit\n");
    exit(1);
  }
  mpz_export(words, &written, -1, sizeof(uint32_t), 0, 0, value);
  while(written<count) words[written++]=0;
}

/*
##################################
print_words
* x: words array
* count: (bits + 31) / 32, number of words in array
*/
void print_words(uint32_t *x, uint32_t count){
    int index;
    for (index=count-1; index>=0; index--){
        //little endian, so we print in reverse order
        printf("%08X", x[index]);
    }
    printf("\n");
}

/*
##################################
compare_words
* para:
* x: words array
* y: words array
* count: (bits + 31) / 32, number of words in array
* return:
* 1 : x > y
* -1: x < y
* 0 : x = y
*/
int compare_words(uint32_t *x, uint32_t *y, uint32_t count){
  int index;
  for (index = count-1; index >= 0; index ++){
    if (x[index] > y[index]) return 1;
    else if (x[index] < y[index]) return -1;
    else continue;
  }
  return 0;
}