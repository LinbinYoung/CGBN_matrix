#include <stdio.h>

typedef struct _rect{
    float height;
    float width;
} Rectangle;

float area(Rectangle rect){
    return rect.height * rect.width;
}