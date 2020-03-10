#1. Basic Library

from ctypes import *
import numpy as np
_cuda_lib = CDLL('./libcompute.so')

#2. Parameters

TPI = 32
CPH_BITS = 2048
COUNT = 2000

#3. Define Big Number

class CGBN_MEM_T:
    def __init__(self, n):
        self._size = n
        self._limbs = np.random.bytes(n)
    def random_value(self, seed):
        np.random.seed(seed)
        self._limbs = np.random.bytes(self._size)
        return int.from_bytes(self._limbs, byteorder='little')

#4. Define Input

class INSTANCE(object):
    def __init__(self, bits, tpi, count, compute_type):
        self._comtype = c_int(compute_type)
        self._bits = c_int(bits)
        self._byte = (bits + 7)//8
        self._tpi = c_int(tpi)
        self._count = c_int(count)
        self.x0 = []
        self.x1 = []
        self.res = create_string_buffer(count*((bits + 7)//8))
        self.num = (c_int32 * 1)()
    def randomInit(self):
        instan = CGBN_MEM_T(self._byte)
        self.num[0] = instan.random_value(0)
        for i in range(self._count.value):
            self.x0.append(instan.random_value(i))
            self.x1.append(instan.random_value(i+99))

#5. Prepare Command Parameters

instan = INSTANCE(CPH_BITS, TPI, COUNT, 3)
instan.randomInit()

c_count = c_int32(len(instan.x0))
print (c_count.value)

array_t_0 = c_int32 * len(instan.x0)
input_0 = array_t_0(*instan.x0)
array_t_1 = c_int32 * len(instan.x1)
input_1 = array_t_1(*instan.x1)
array_t_2 = c_int32 * len(instan.num)
input_2 = array_t_2(*instan.num)

_cuda_lib.run_gpu(instan._comtype, instan._tpi, instan._bits, input_0, input_1, input_2, instan.res, instan._count)
