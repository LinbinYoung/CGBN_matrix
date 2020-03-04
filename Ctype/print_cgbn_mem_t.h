#include<iostream>

template<uint32_t bits>
struct cgbn_mem_t {
  public:
  uint32_t _limbs[(bits+31)/32];
};

void print_words(uint32_t *x, uint32_t count){
    int index;
    for (index=count-1; index>=0; index--){
        //little endian, so we print in reverse order
        printf("%08X", x[index]);
    }
    printf("\n");
}
