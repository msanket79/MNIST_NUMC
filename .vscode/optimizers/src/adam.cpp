#include<numC/npArrayCpu.hpp>
#include<numC/npFunctions.hpp>

#include<optimizers/adam.hpp>
#include<cmath>
// AdamOptimizer(const float learning_rate=0.001,const float beta1=0.9,const float beta2=0.999,const float epsilon=1e-8);
// AdamOptimizer(AdamOptimizer &A);
// void operator=(AdamOptimizer &A);

// void step(np::ArrayCpu<float>&param,np::ArrayCpu<float>&grad);
AdamOptimizer::AdamOptimizer(const float learning_rate=0.001,const float beta1=0.9,const float beta2=0.999,const float epsilon=1e-8){
    this->learing_rate=learning_rate;
    this->beta1=beta1;
    this->beta2=beta2;
    this->epsilon=epsilon;
}
AdamOptimizer::AdamOptimizer(AdamOptimizer&A){
    this->learing_rate=A.learing_rate;
    this->beta1=A.beta1;
    this->beta2=A.beta2;
    this->epsilon=A.epsilon;
    this->m=A.m;
    this->v=A.v;
    this->t=A.t;
}
void AdamOptimizer::operator=(AdamOptimizer&A){
    this->learing_rate=A.learing_rate;
    this->beta1=A.beta1;
    this->beta2=A.beta2;
    this->epsilon=A.epsilon;
    this->m=A.m;
    this->v=A.v;
    this->t=A.t;
}
void AdamOptimizer::step(np::ArrayCpu<float>&param,np::ArrayCpu<float>&grad){
    // time factor
    ++this->t;

    // for first moment
    this->m=this->m*beta1+grad*(1-this->beta1);
    auto mt=this->m/(1-powf(this->beta1,static_cast<float>(this->t)));

    //for velocity
    this->v=this->v*beta2+grad*(1-this->beta2);
    auto vt=this->v/(1-powf(this->beta2,static_cast<float>(this->t)));
    param=param-((mt*this->learing_rate)/(np::sqrt(vt)+this->epsilon));


}