#include "interface.h"

int main(){
    int a = 1;
    int b = 2;
    int c = 3;
    int *res = (int*)malloc(sizeof(int)*c);
    interface(a,b,c,res,2);
    for (int i = 0; i < c; i ++){
        cout << res[i] << endl;
    }
    free(res);
    return 0;
}