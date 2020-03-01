#include<iostream>
#include<random>

using namespace std;

class TaskBase{
    public:
        int _numa;
        int _numb;
        int _count;
        int *arr_1;
        int *arr_2;
        TaskBase(int a = 0, int b = 0, int count = 10): _numa(a), _numb(b), _count(count){cout << "I am the Base" << endl;}
        virtual ~TaskBase(){cout << "Free Memory from the Base" << endl;}
        virtual void initialized(int *d){cout << "Initialized from the Base" << endl;}
        virtual int compute(){cout << "Calling compute from Base" << endl; return 0;};
};

class GPUTask: public TaskBase{
    public:
        GPUTask(int a, int b, int count):TaskBase(a, b, count){
            this->arr_1 = (int*)malloc(sizeof(int)*this->_count);
            this->arr_2 = (int*)malloc(sizeof(int)*this->_count);
            for (int index = 0; index < this->_count; index ++){
                this->arr_1[index] = rand();
                this->arr_2[index] = rand();
            }
            cout << "Constructor from the GPU" << endl;
        }
        ~GPUTask(){
            free(this->arr_1);
            free(this->arr_2);
            cout << "Free Memory from the GPU" << endl;
        }
        int compute(){
            return this->_numa + this->_numb;
        }
        void initialized(int* c){
            cout << "Calling initialized method from GPU" << endl;
            for (int index = 0; index < this->_count; index ++){
                c[index] = this->arr_1[index] + this->arr_2[index];
            }
        }
};

class CPUTask: public TaskBase{
    public:
        CPUTask(int a, int b, int count):TaskBase(a, b, count){
            this->arr_1 = (int*)malloc(sizeof(int)*this->_count);
            this->arr_2 = (int*)malloc(sizeof(int)*this->_count);
            for (int index = 0; index < this->_count; index ++){
                this->arr_1[index] = rand();
                this->arr_2[index] = rand();
            }
            cout << "Constructor from the CPU" << endl;
        }
        ~CPUTask(){
            free(this->arr_1);
            free(this->arr_2);
            cout << "Free Memory from the CPU" << endl;
        }
        int compute(){
            return this->_numa - this->_numb;
        }
        void initialized(int* c){
            cout << "Calling initialized method from CPU" << endl;
            for (int index = 0; index < this->_count; index ++){
                c[index] = this->arr_1[index] + this->arr_2[index];
            }
        }
};

class FPGATask: public TaskBase{
    public:
        FPGATask(int a, int b, int count):TaskBase(a, b, count){
            this->arr_1 = (int*)malloc(sizeof(int)*this->_count);
            this->arr_2 = (int*)malloc(sizeof(int)*this->_count);
            for (int index = 0; index < this->_count; index ++){
                this->arr_1[index] = rand();
                this->arr_2[index] = rand();
            }
            cout << "Constructor from the FPGA" << endl;
        }
        ~FPGATask(){
            free(this->arr_1);
            free(this->arr_2);
            cout << "Free Memory from the FPGA" << endl;
        }
        int compute(){
            return this->_numa * this->_numb;
        }
        void initialized(int* c){
            cout << "Calling initialized method from FPGA" << endl;
            for (int index = 0; index < this->_count; index ++){
                c[index] = this->arr_1[index] + this->arr_2[index];
            }
        }
};

extern "C"{
    void interface(int a, int b, int c, void *d, int device);
}