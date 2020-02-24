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


/**************************************************************************
 * Addition: scalar + scalar
 **************************************************************************/
template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_add(xmp_tester<tpi, bits>::x_instance_t *instances) {
  //int32_t LOOPS=LOOP_COUNT(bits, xt_add);
  bn_t    x0, x1, r;
  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));
  _env.add(r, x0, x1);
  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_add_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  if(instance>=count)
    return;
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_add(instances);
}


/**************************************************************************
 * Addition: scalar + BigNum
 **************************************************************************/
 template<uint32_t tpi, uint32_t bits>
 __device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_addui(xmp_tester<tpi, bits>::x_instance_t *instances) {
   //int32_t LOOPS=LOOP_COUNT(bits, xt_add);
   bn_t    x0, num, r;
   _env.load(x0, &(instances[_instance].x0));
   _env.load(num, &(instances[_instance].num));
   _env.add(r, x0, num);
   _env.store(&(instances[_instance].r), r);
 }
 
 template<uint32_t tpi, uint32_t bits>
 __global__ void x_test_addui_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
   uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
   if(instance>=count)
     return;
   xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
   tester.x_test_addui(instances);
 }

/**************************************************************************
 * Multiplication: scalar * num
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_mul(xmp_tester<tpi, bits>::x_instance_t *instances) {
  //int32_t   LOOPS=LOOP_COUNT(bits, xt_mul);
  bn_t      x0, num, r;
  bn_wide_t w;

  _env.load(x0, &(instances[_instance].x0));
  _env.load(num, &(instances[_instance].num));

  _env.mul_wide(w, x0, num);
  _env.set(r, w._low);
  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_mul_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_mul(instances);
}