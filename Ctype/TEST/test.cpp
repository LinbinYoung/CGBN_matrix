#include <iostream>
#include <cstring>

using namespace std;

class DataInstan{
    public:
    int *x;
    int *y;
    DataInstan(int size){
        this->x = (int*)malloc(sizeof(int)*size);
        this->y = (int*)malloc(sizeof(int)*size);
    }
    ~DataInstan(){
        free(this->x);
        free(this->y);
        cout << "Memory Free" << endl;
    }
};

extern "C"{
    void interface(uint32_t *x, uint32_t *y, char* z, int count, int size){
        DataInstan* acc = new DataInstan(count);
        /*
            Initialized Data Block
        */
        for (int i = 0; i < count; i ++) memcpy(acc->x + i, x + i, size);
        for (int j = 0; j < count; j ++) memcpy(acc->y + j, y + j, size);
        /*
            After computation
        */
        memcpy(z, acc->x, size*count);
        delete acc;
    }
}

int main(){
    int x[10] = {1,2,3,4,5,6,7,8,9,10};
    int y[10] = {1,2,3,4,5,6,7,8,9,10};
    char *z = (char*)malloc(sizeof(char)*40);
    interface((uint32_t*)x, (uint32_t*)y, z, 10, 4);
    uint32_t temp;
    printf("Linbin YANG");
    for (int i = 0; i < 40; i=i+4){
        memcpy(&temp, z+i, 4);
        printf("%d", temp);
    }
    return 0;
}