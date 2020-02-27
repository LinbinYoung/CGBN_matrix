# 接口说明
还未测试，有待进一步讨论完善。


### 输入和输出数据
数据类型：
```c++
enum struct CCmode{
  gpu,
  cpu,
  fpga
};
```
输入：
```c++
typedef struct input_instance{
    public：
      cgbn_mem_t<bits> *x0;
      cgbn_mem_t<bits> *x1;
      cgbn_mem_t<bits> num;
      Cmode type; 
};
```
输出：
```c++
typedef struct mem_results{
    public:
      cgbn_mem_t<bits> *r;
      Cmode type;
}
```

### 计算接口
可调整参数：算子操作类型，TPI，数据大小，数据总量
```c++
void ComputeInterface(test_t operation, uint32_t tpi, uint32_t size, void *input, void *output, uint32_t count)；
```
