#ifndef LAYER_HPP
#define LAYER_HPP

#include<numC/npArrayCpu.hpp>

// layers prototype
class Layer{
    public:
    // to store any sort of outer values
    np::ArrayCpu<float>cache;
    
    // forward passs------------------------------------------------------------------
    virtual np::ArrayCpu<float> forward( np::ArrayCpu<float>&X,const std::string&mode="train")=0;
    virtual np::ArrayCpu<float> operator()( np::ArrayCpu<float>&X,const std::string&mode="train")=0;

    //backward pass
    virtual np::ArrayCpu<float> backward(np::ArrayCpu<float>&dOut)=0;
    
    

};
#endif
