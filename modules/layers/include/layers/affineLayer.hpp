#ifndef AFFINELAYER_HPP
#define AFFINELAYER_HPP


#include<layers/layer.hpp>
#include<numC/npArrayCpu.hpp>


class AffineLayer:public Layer{
    public:
        // weights
        np::ArrayCpu<float>w,b;
        np::ArrayCpu<float>dw,db;
        AffineLayer(int in_features=1,int out_features=1);

        AffineLayer(const AffineLayer &L);

        void operator=(const AffineLayer &L);

        // forward pass---------------------------------------------------------------------------
        np::ArrayCpu<float> forward(np::ArrayCpu<float>&X,const std::string &mode="train") override;
        np::ArrayCpu<float> operator()( np::ArrayCpu<float>&X,const std::string &mode="train") override;

        // backward pass--------------------------------------------------------------------------
        np::ArrayCpu<float> backward(np::ArrayCpu<float>&dOut) override;


        

};

#endif