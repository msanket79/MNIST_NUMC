#ifndef DROPOUT_HPP
#define DROPOUT_HPP

#include<numC/npArrayCpu.hpp>

#include<layers/layer.hpp>

class DropoutLayer:public Layer{
    public:
    float p_keep;
    DropoutLayer(const float p_keep=1);
    DropoutLayer(const DropoutLayer &L);
    void operator=(const DropoutLayer &L);
    np::ArrayCpu<float> forward(np::ArrayCpu<float>&X,const std::string&mode="train");
    np::ArrayCpu<float> operator()(np::ArrayCpu<float>&X,const std::string&mode="train");
    np::ArrayCpu<float> backward(np::ArrayCpu<float>&dOut);


};
#endif