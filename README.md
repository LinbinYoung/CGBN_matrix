# 接口说明
注意检查一下是否存在内存泄漏的问题。
### 输入和输出数据

####输入：
```c++
template<uint32_t bits>
class DataBase{
  public:
    cgbn_mem_t<bits> *x0;
    cgbn_mem_t<bits> *x1;
    cgbn_mem_t<bits> num;
    DataBase(uint32_t count):{}
    virtual ~DataBase(){}
};
```
#####GPU输入数据：
```c++
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
```

#####CPU输入数据：
```c++
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
```

####输出：
```c++
template<uint32_t bits>
class ResultBase{
  public:
    cgbn_mem_t<bits> *r;
    DataBase(uint32_t count):{}
    virtual ~DataBase(){}
};
```
#####GPU输出数据：
```c++
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
```
#####CPU输出数据：
```c++
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
```

### 计算接口
可调整参数：算子操作类型，TPI，数据大小，数据总量
```c++
void ComputeInterface(test_t operation, uint32_t tpi, uint32_t size, void *input, void *output, uint32_t count)；
```
