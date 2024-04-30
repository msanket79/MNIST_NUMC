#ifndef ADAM_HPP
#define ADAM_HPP
#include<numC/npArrayCpu.hpp>
#include<layers/layer.hpp>

class AdamOptimizer:public Layer
{
    public:
        float learing_rate,beta1,beta2,epsilon;
        int t;

        np::ArrayCpu<float>m,v;
        AdamOptimizer(const float learning_rate=0.001,const float beta1=0.9,const float beta2=0.999,const float epsilon=1e-8);
        AdamOptimizer(AdamOptimizer &A);
        void operator=(AdamOptimizer &A);

        void step(np::ArrayCpu<float>&param,np::ArrayCpu<float>&grad);


};
#endif