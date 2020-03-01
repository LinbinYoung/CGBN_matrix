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