/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/

#include "common.h"

template<uint32_t tpi, uint32_t bits>
class TaskBase{
  public:
    typedef cgbn_context_t<tpi>                context_t;
    typedef cgbn_env_t<context_t, bits>        env_t;
    typedef typename env_t::cgbn_t             bn_t;
    typedef typename env_t::cgbn_local_t       bn_local_t;
    typedef typename env_t::cgbn_wide_t        bn_wide_t;
    typedef typename env_t::cgbn_accumulator_t bn_accumulator_t;

    context_t _context;
    env_t     _env;
    int32_t   _instance;
    __device__ __forceinline__ TaskBase(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance) : _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {}
    static __host__ void AcceptData(gmp_randstate_t state, DataBase<bits> *ins, uint32_t count){
      mpz_t         value;
      mpz_t         fixedvalue;
      mpz_init(value);
      mpz_init(fixedvalue);
      mpz_urandomb(fixedvalue, state, bits);
      from_mpz(ins->num._limbs, bits/32, fixedvalue);
      for (int index = 0; index < count; index ++){
        mpz_urandomb(value, state, bits);
        from_mpz(ins->x0[index]._limbs, bits/32, value);
        mpz_urandomb(value, state, bits);
        from_mpz(ins->x1[index]._limbs, bits/32, value);
      }
      mpz_clear(fixedvalue);
      mpz_clear(value);
    }
};

template<uint32_t tpi, uint32_t bits>
class GPUTask : public TaskBase<tpi, bits> {
  public:
    __device__ __forceinline__ GPUTask(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance) : TaskBase<tpi, bits>(monitor, report, instance) {}  

    __device__ __forceinline__  void x_test_add(GPU_Data<bits> *instances, GPU_result<bits> *res);
    __device__ __forceinline__  void x_test_addui(GPU_Data<bits> *instances, GPU_result<bits> *res);
    __device__ __forceinline__  void x_test_mul(GPU_Data<bits> *instances, GPU_result<bits> *res);
};

#include "GPU.cu"

/*
 * Function: x_run_test
 * Description: Used to run test under different data size and tpi
 * Para:
 *   test_t       : type of operation
 *   x_instance_t : the start address of actual instances
 *   res          : the start address of result array
 *   count       : number of instances
 */
template<uint32_t tpi, uint32_t bits>
void x_run_test(Compute_Type operation, DataBase<bits> *instances, ResultBase<bits> *res, uint32_t count) {
  int threads=128, IPB=threads/tpi, blocks=(count+IPB-1)*tpi/threads;
  printf("Number of threads in block %d\n", threads);
  printf("Number of instances can be processed %d\n", IPB);
  printf("Number of blocks %d\n", blocks);
  if(operation==xt_add){
      x_test_add_kernel<tpi, bits><<<blocks, threads>>>((GPU_Data<bits>*)instances, (GPU_result<bits>*)res, count); 
      CUDA_CHECK(cudaDeviceSynchronize());
  }
  else if (operation==xt_addui){
      x_test_addui_kernel<tpi, bits><<<blocks, threads>>>((GPU_Data<bits>*)instances, (GPU_result<bits>*)res, count); 
      CUDA_CHECK(cudaDeviceSynchronize());
  }
  else if (operation==xt_mul){
      x_test_mul_kernel<tpi, bits><<<blocks, threads>>>((GPU_Data<bits>*)instances, (GPU_result<bits>*)res, count); 
      CUDA_CHECK(cudaDeviceSynchronize());
  }
  else {
    printf("Unsupported operation -- needs to be added to x_run_test<...> in xmp_tester.cu\n");
    exit(1);
  }
}

/*
 * Function: x_run_test
 * Description: Used to run test under different data size and tpi
 * Para:
 *   test_t      : type of operation
 *   instances       : the start address of actual instances
 *   res_pool        : the start address of result pool
 *   count       : number of instances
 *   repetitions : repetitions for each task
 */
template<uint32_t tpi, uint32_t bits>
void x_run_test(Compute_Type operation, void *instances, void *res_cpu, uint32_t count) {

  /*
    1. Allocate memory for input and output data
  */

  GPU_Data<bits> *input_gpuins = new GPU_Data<bits>(count);
  CPU_Data<bits> *input_cpuins = new CPU_Data<bits>(count);
  GPU_result<bits> *output_gpu = new GPU_result<bits>(count);
  
  /*
    2. Initialized data
  */

  CUDA_CHECK(cudaMemcpy(input_gpuins->x0, ((GPU_Data<bits>*)instances)->x0, sizeof(cgbn_mem_t<bits>)*count, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(input_gpuins->x1, ((GPU_Data<bits>*)instances)->x1, sizeof(cgbn_mem_t<bits>)*count, cudaMemcpyHostToDevice));
  /*
   3. Start compute on GPU
  */
  Timer gpu;
  x_run_test<tpi, bits>(operation, (GPU_Data<bits> *)input_gpuins, (GPU_result<bits> *)output_gpu, count);
  printf("GPU, computation: %.31f s\n", gpu.stop());
  CUDA_CHECK(cudaMemcpy(((CPU_result<bits>*)res_cpu)->r, output_gpu->r, sizeof(cgbn_mem_t<bits>)*count, cudaMemcpyDeviceToHost)); //copy results back to memory
  /*
    4. Task finished, free memory
  */
  delete input_gpuins;
  delete input_cpuins;
  delete output_gpu;
  return;
}

template<uint32_t tpi, uint32_t bits>
void* Data_Generator(gmp_randstate_t state, uint32_t count){
  if(!supported_tpi_size(tpi, bits)){
      return NULL;
  }
  DataBase<bits>* instance = new CPU_Data<bits>(count);
  TaskBase<tpi, bits>::AcceptData(state, instance, count);
  return (void *) instance;
}

extern "C"{
  /*
  * Function: run_gpu
  * Description: Used to run test under different data size and tpi
  * Para:
  *   test_t      : type of operation
  *   tpi         : number of threads for each instance
  *   size        : size of each instance
  *   input       : the input
  *   output      : the output
  *   count       : number of instance
  */
  void run_gpu(Compute_Type operation, uint32_t tpi, uint32_t size, void *input, void *output, uint32_t count) {
    if(!supported_tpi_size(tpi, size)) {
      printf("Unsupported tpi and size -- needs to be added to x_run_test in xmp_tester.cu\n");
      exit(1);
    }
    if(tpi==4 && size==128)
      x_run_test<4, 128>(operation, input, output, count);
    else if(tpi==4 && size==256)
      x_run_test<4, 256>(operation, input, output, count);
    else if(tpi==8 && size==256)
      x_run_test<8, 256>(operation, input, output, count);
    else if(tpi==4 && size==512)
      x_run_test<4, 512>(operation, input, output, count);
    else if(tpi==8 && size==512)
      x_run_test<8, 512>(operation, input, output, count);
    else if(tpi==16 && size==512)
      x_run_test<16, 512>(operation, input, output, count);
    else if(tpi==8 && size==1024)
      x_run_test<8, 1024>(operation, input, output, count);
    else if(tpi==16 && size==1024)
      x_run_test<16, 1024>(operation, input, output, count);
    else if(tpi==32 && size==1024)
      x_run_test<32, 1024>(operation, input, output, count);
    else if(tpi==8 && size==2048)
      x_run_test<8, 2048>(operation, input, output, count);
    else if(tpi==16 && size==2048)
      x_run_test<16, 2048>(operation, input, output, count);
    else if(tpi==32 && size==2048)
      {printf("call run_gpu\n"); x_run_test<32, 2048>(operation, input, output, count);}
    else if(tpi==16 && size==3072)
      x_run_test<16, 3072>(operation, input, output, count);
    else if(tpi==32 && size==3072)
      x_run_test<32, 3072>(operation, input, output, count);
    else if(tpi==16 && size==4096)
      x_run_test<16, 4096>(operation, input, output, count);
    else if(tpi==32 && size==4096)
      x_run_test<32, 4096>(operation, input, output, count);
    else if(tpi==32 && size==5120)
      x_run_test<32, 5120>(operation, input, output, count);
    else if(tpi==32 && size==6144)
      x_run_test<32, 6144>(operation, input, output, count);
    else if(tpi==32 && size==7168)
      x_run_test<32, 7168>(operation, input, output, count);
    else if(tpi==32 && size==8192)
      x_run_test<32, 8192>(operation, input, output, count);
    else {
      printf("internal error -- tpi/size -- needs to be added to x_run_test in xmp_tester.cu\n");
      exit(1);
    }
  }
}

#ifndef INSTANCES
#define INSTANCES 2000
#endif

#ifndef DATA_SIZE
#define DATA_SIZE 2048
#endif

#ifndef TPI
#define TPI 32
#endif

int main() {
  gmp_randstate_t  state;
  void             *input_data;
  void             *output_data;

  gmp_randinit_default(state);

  /*
   Start the task
  */
  if(!supported_size(DATA_SIZE)) printf("... %d ... invalid test size ...\n", DATA_SIZE);
  printf("... generating data ...\n");
  input_data=Data_Generator<TPI, DATA_SIZE>(state, INSTANCES);
  ResultBase<DATA_SIZE>* result = new CPU_result<DATA_SIZE>(INSTANCES);
  output_data = (void*)result; //allocate memory for result
  if(!supported_tpi_size(TPI, DATA_SIZE))return 0;
  printf("... %s %d:%d ... ", actual_compute_name(XT_FIRST), DATA_SIZE, TPI); fflush(stdout);
  run_gpu(XT_FIRST, TPI, DATA_SIZE, input_data, output_data, INSTANCES);
  return 0;
}