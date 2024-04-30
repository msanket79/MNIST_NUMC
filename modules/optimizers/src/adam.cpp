    #include<optimizers/adam.hpp>
    #include<numC/npArrayCpu.hpp>
#include<numC/npFunctions.hpp>
    AdamOptimizer::AdamOptimizer(const float learning_rate,const float beta1,const float beta2,const float epsilon){
            this->learning_rate=learning_rate;
            this->beta1=beta1;
            this->beta2=beta2;
            this->epsilon=epsilon;
            this->m=np::zeros<float>(1,1);
            this->v=np::zeros<float>(1,1);
            this->t=0;
        }
        AdamOptimizer::AdamOptimizer(const AdamOptimizer& A) {
            this->learning_rate=A.learning_rate;
            this->beta1=A.beta1;
            this->beta2=A.beta2;
            this->epsilon=A.epsilon;
            this->m=A.m;
            this->v=A.v;
            this->t=A.t;

        }
        void AdamOptimizer::operator=(const AdamOptimizer &A){
            this->learning_rate=A.learning_rate;
            this->beta1=A.beta1;
            this->beta2=A.beta2;
            this->epsilon=A.epsilon;
            this->m=A.m;
            this->v=A.v;
            this->t=A.t;
        }
        void AdamOptimizer::step(np::ArrayCpu<float>&param,np::ArrayCpu<float>&grad){
            ++this->t;
            this->m=(this->m*this->beta1)+grad*(1-this->beta1);
            auto mt=this->m / (1-powf( this->beta1, static_cast<float>(this->t) ) );
            this->v=( this->v*this->beta2 ) +np::square(grad)*(1-this->beta2);
            auto vt=this->v / (1-powf( this->beta2, static_cast<float>(this->t) ) );
            param = param - (( mt*this->learning_rate )/( np::sqrt(vt)+ this->epsilon ));

        }