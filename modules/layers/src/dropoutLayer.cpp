#include<numC/npArrayCpu.hpp>
#include<numC/npRandom.hpp>

#include<layers/dropoutlayer.hpp>



DropoutLayer::DropoutLayer(const float p_keep=1){
    this->p_keep=p_keep;

}
DropoutLayer::DropoutLayer(const DropoutLayer &L){
    this->p_keep=L.p_keep;
    this->cache=L.cache;

}
void DropoutLayer::operator=(const DropoutLayer&L){
    this->p_keep=L.p_keep;
    this->cache=L.cache;

}
np::ArrayCpu<float> DropoutLayer::forward(np::ArrayCpu<float>&X,const std::string &mode){


    if(mode=="train"){
        this->cache=(np::Random::rand<float>(X.rows,X.cols)< this->p_keep)/this->p_keep;
        return this->cache*X;
    }
    return X;

}
np::ArrayCpu<float>DropoutLayer::operator()(np::ArrayCpu<float>&X,const std::string &mode){
    if(mode=="train"){
        this->cache=(np::Random::rand<float>(X.rows,X.cols)<this->p_keep)/p_keep;
        return this->cache*X;
    }
}

np::ArrayCpu<float>DropoutLayer::backward(np::ArrayCpu<float>&dOut){
    return this->cache*dOut;
}