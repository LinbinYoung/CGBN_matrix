#include "interface.h"

extern "C"{
    /*
    ###########################
    interface
    * a : para 1
    * b : para 2
    * c : para 3
    * d : return arr
    */
    void interface(int a, int b, int c, void *d, int device){
        /*
         0 - CPU
         1 - GPU
         2 - FPGA
        */
        TaskBase *task;
        if (device == 0){
            task = new CPUTask(a, b, c);
        }else if (device == 1){
            task = new GPUTask(a, b, c);
        }else if (device == 2){
            task = new FPGATask(a, b, c);
        }
        task->initialized((int*)d);
        delete task;
    }
}

int main(){
    int a = 1;
    int b = 2;
    int c = 3;
    int *res = (int*)malloc(sizeof(int)*c);
    interface(a,b,c,(void*)res,2);
    for (int i = 0; i < c; i ++){
        cout << res[i] << endl;
    }
    free(res);
    return 0;
}