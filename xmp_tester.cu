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
#include "gpu_support.h"

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

template<uint32_t tpi, uint32_t bits>
class GPU_task {
  public:
  typedef cgbn_context_t<tpi>                context_t;
  typedef cgbn_env_t<context_t, bits>        env_t;
  typedef typename env_t::cgbn_t             bn_t;
  typedef typename env_t::cgbn_local_t       bn_local_t;
  typedef typename env_t::cgbn_wide_t        bn_wide_t;
  typedef typename env_t::cgbn_accumulator_t bn_accumulator_t;
  
  context_t _context;
  env_t     _env;
  int32_t   _instance; //id of instance
  
  __device__ __forceinline__ GPU_task(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance) : _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {}  

  static __host__ void AcceptData(gmp_randstate_t state, input_instance &ins, uint32_t count) {
    assert(ins.type == Cmode::cpu);
    mpz_t         value;
    mpz_t         fixedvalue;
    mpz_init(value);
    mpz_init(fixedvalue);
    mpz_urandomb(fixedvalue, state, bits);
    from_mpz(ins.num._limbs, bits/32, fixedvalue);
    for (int index = 0; index < count; index ++){
      mpz_urandomb(value, state, bits);
      from_mpz(ins.x0[index]._limbs, bits/32, value);
      mpz_urandomb(value, state, bits);
      from_mpz(ins.x1[index]._limbs, bits/32, value);
    }
    mpz_clear(fixedvalue);
    mpz_clear(value);
}

  __device__ __forceinline__ void x_test_add(x_instance_t *instances, mem_results *res);
  __device__ __forceinline__ void x_test_addui(x_instance_t *instances, mem_results *res);
  __device__ __forceinline__ void x_test_mul(x_instance_t *instances, mem_results *res);
};

#include "xmp_tests.cu"

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
void x_run_test(test_t operation, input_instance<bits> *instances, mem_results<bits> *res, uint32_t count) {
  int threads=128, IPB=threads/tpi, blocks=(count+IPB-1)*tpi/threads;
  if(operation==xt_add) 
    x_test_add_kernel<tpi, bits><<<blocks, threads>>>(instances, res, count);
  else if (operation==xt_addui)
    x_test_addui_kernel<tpi, bits><<<blocks, threads>>>(instances, res, count);
  else if (operation==xt_mul)
    x_test_mul_kernel<tpi, bits><<<blocks, threads>>>(instances, res, count);
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
void x_run_test(test_t operation, void *instances, void *res_cpu, uint32_t count) {

  /*
    1. Allocate memory for input and output data
  */

  input_instance<bits> gpuInstances(count, Cmode::gpu);
  input_instance<bits> cpuInstances(count, Cmode::cpu);
  mem_results<bits> gpuResult(count, Cmode::gpu);

  input_instance *input_gpuins = &gpuInstances;
  input_instance *input_cpuins = &cpuInstances;
  mem_results *output_gpu = &gpuResult;
  
  /*
    2. Initialized data
  */

  CUDA_CHECK(cudaMemcpy(input_gpuins, instances, sizeof(input_instance<bits>)*count, cudaMemcpyHostToDevice));

  /*
   3. Start compute on GPU
  */
  Timer gpu;
  x_run_test<tpi, bits>(operation, (input_instance<bits> *)input_gpuins, (mem_results<bits> *)output_gpu, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("GPU, computation: %.31f s\n", gpu.stop());
  CUDA_CHECK(cudaMemcpy(res_cpu, output_gpu, sizeof(mem_results<bits>)*count, cudaMemcpyDeviceToHost)); //copy results back to memory
  /*
    4. Task finished
  */
  return;
}


/*
 * Function: gpu_run_interface
 * Description: Used to run test under different data size and tpi
 * Para:
 *   test_t      : type of operation
 *   tpi         : number of threads for each instance
 *   size        : size of each instance
 *   input       : the input
 *   output      : the output
 *   count       : number of instance
 */
void x_run_test(test_t operation, uint32_t tpi, uint32_t size, void *input, void *output, uint32_t count) {
  if(!x_supported_tpi_size(tpi, size)) {
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
    x_run_test<32, 2048>(operation, input, output, count);
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

template<uint32_t bits>
void* Data_Generator(gmp_randstate_t state, uint32_t tpi, uint32_t size, uint32_t count){
  input_instance<bits> instance(count, Cmode::cpu);
  if(size==128)
    GPU_task<32, 128>::AcceptData(state, instance, count);
  else if(size==256)
    GPU_task<32, 256>::AcceptData(state, instance, count);
  else if(size==512)
    GPU_task<32, 512>::AcceptData(state, instance, count);
  else if(size==1024)
    GPU_task<32, 1024>::AcceptData(state, instance, count);
  else if(size==2048)
    GPU_task<32, 2048>::AcceptData(state, instance, count);
  else if(size==3072)
    GPU_task<32, 3072>::AcceptData(state, instance, count);
  else if(size==4096)
    GPU_task<32, 4096>::AcceptData(state, instance, count);
  else if(size==5120)
    GPU_task<32, 5120>::AcceptData(state, instance, count);
  else if(size==6144)
    GPU_task<32, 6144>::AcceptData(state, instance, count);
  else if(size==7168)
    GPU_task<32, 7168>::AcceptData(state, instance, count);
  else if(size==8192)
    GPU_task<32, 8192>::AcceptData(state, instance, count);
  else {
    printf("Unsupported size -- needs to be added to x_generate_data in xmp_tester.cu\n");
    exit(1);
  }
  return (void *) &instance;
}

void ComputeInterface(test_t operation, uint32_t tpi, uint32_t size, void *input, void *output, uint32_t count){
  x_run_test(XT_FIRST, TPI, DATA_SIZE, input_data, output_data, INSTANCES);
  return;
}

#ifndef INSTANCES
#define INSTANCES 200000
#endif

#ifndef DATA_SIZE
#define DATA_SIZE 2048
#endif

#ifndef TPI
#define TPI 4
#endif

int main() {
  gmp_randstate_t  state;
  void             *input_data;
  void             *output_data;

  gmp_randinit_default(state);

  /*
   Start the task
  */
  if(!supported_size(DATA_SIZE)) printf("... %d ... invalid test size ...\n", sizes[index]);
  printf("... generating data ...\n");
  input_data=x_generate_data(state, 32, DATA_SIZE, INSTANCES);
  mem_results<DATA_SIZE> result(INSTANCES);
  output_data = (void*)&result; //allocate memory for result
  if(!x_supported_tpi_size(TPI, DATA_SIZE)) continue;
  printf("... %s %d:%d ... ", test_name(XT_FIRST), DATA_SIZE, TPI); fflush(stdout);
  ComputeInterface(XT_FIRST, TPI, DATA_SIZE, input_data, output_data, INSTANCES);
  return 0;
}