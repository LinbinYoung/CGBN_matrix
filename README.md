### Bignum Parallel Computing on GPU

可调整参数：算子操作类型，TPI，数据大小，数据总量

C++ 接口
```c++
void run_gpu(uint32_t operation, uint32_t tpi, uint32_t size, uint32_t *input_0, uint32_t *input_1, uint32_t *input_2, void *output_data, uint32_t count);
```
Python 接口

使用Ctype进行Python和C++之间的交互

```python
_cuda_lib.run_gpu(instan._comtype, instan._tpi, instan._bits, input_0, input_1, input_2, instan.res, instan._count)
```
