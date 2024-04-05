#ifndef RELULAYER_HPP
#define RELULAYER_HPP

#include<numC/npArrayCpu.hpp>
#include<numC/npFunctions.hpp>
#include<layers/layer.hpp>

class ReluLayer:public Layer{
    public:
    ReluLayer();
    ReluLayer(const ReluLayer&L);
    void operator=(const ReluLayer&L);
    np::ArrayCpu<float> forward(np::ArrayCpu<float>&X,const std::string &mode="train");
    np::ArrayCpu<float> operator()(np::ArrayCpu<float>&X,const std::string &mode="train");
    np::ArrayCpu<float> backward(np::ArrayCpu<float>&dOut);


};


#endif