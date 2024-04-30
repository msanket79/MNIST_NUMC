#include<numC/npArrayCpu.hpp>
#include<numC/npFunctions.hpp>

#include<layers/reluLayer.hpp>

    // np::ArrayCpu<float> forward(np::ArrayCpu<int>&X,const std::string &mode);
    // np::ArrayCpu<float> operator()(np::ArrayCpu<int>&X,const std::string &mode);
    // np::ArrayCpu<float> backward(np::ArrayCpu<int>&dOut);

ReluLayer::ReluLayer(){
    ;
}
ReluLayer::ReluLayer(const ReluLayer&L){
    this->cache=L.cache;
}
void ReluLayer::operator=(const ReluLayer&L){
    this->cache=L.cache;
}
np::ArrayCpu<float> ReluLayer::forward(np::ArrayCpu<float>&X,const std::string &mode){
    if(mode=="train"){
        this->cache=X;
    }
    
    auto out= np::maximum<float>(X,0.0f);
    return X;

}
np::ArrayCpu<float> ReluLayer::operator()(np::ArrayCpu<float>&X,const std::string &mode){
    return this->forward(X,mode);

}
np::ArrayCpu<float> ReluLayer::backward(np::ArrayCpu<float>&dOut){
    auto dX=dOut*(this->cache>0);
    return dX;
}
