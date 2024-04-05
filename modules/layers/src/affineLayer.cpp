#include<numC/npArrayCpu.hpp>
#include<numC/npFunctions.hpp>
#include<numC/npRandom.hpp>
#include<layers/affineLayer.hpp>


AffineLayer::AffineLayer(int in_features=1,int out_features=1){
    // xavier's initialisation
    this->w=np::Random::randn<float>(in_features,out_features)*sqrtf(2/in_features);
    this->b=np::zeros<float>(1,out_features);


}
AffineLayer::AffineLayer(const AffineLayer &L){
    this->w=L.w;
    this->b=L.b;
    this->dw=L.dw;
    this->db=L.db;
    this->cache=L.cache;
}

void AffineLayer::operator=(const AffineLayer &L){
    this->w=L.w;
    this->b=L.b;
    this->dw=L.dw;
    this->db=L.db;
    this->cache=L.cache;
}
np::ArrayCpu<float> AffineLayer::forward(np::ArrayCpu<float>&X,const std::string &mode){
     auto Z=X.dot(this->w);
     if(mode=="train")
        this->cache=X;
    return Z;
}
np::ArrayCpu<float> AffineLayer::operator()( np::ArrayCpu<float>&X,const std::string &mode){
     auto Z=X.dot(this->w);
     if(mode=="train")
        this->cache=X;
    return Z;    
}
np::ArrayCpu<float> AffineLayer::backward(np::ArrayCpu<float>&dOut){
    this->db=dOut.sum(0);
    this->dw=this->cache.Tdot(dOut);
    auto dX=dOut.dotT(this->w);
    return dX;
}
