// numC includes
#include "numC/npArrayCpu.hpp"
#include "numC/npFunctions.hpp"

// loss function
#include "lossFunctions/softmax.hpp"

//std
#include<vector>

     np::ArrayCpu<float> SoftmaxLoss::computeLoss( np::ArrayCpu<float>&x, np::ArrayCpu<int>&y){
         auto shifted_x = x - x.max(1);

         auto exp_x = np::exp(shifted_x);

         auto denom = exp_x.sum(1);

         auto scores = exp_x / denom;
         scores = scores + 1e-10;
         //std::cout <<"\n scores\n" << scores;
         //std::cout << "\n scores(N,y) \n" << scores.get(np::arange<int>(x.rows), y);
         //std::cout << y;
         auto loss = (-np::log(scores.get(np::arange<int>(x.rows), y))).sum() / x.rows;
        return loss;

    }
    std::vector<np::ArrayCpu<float>>  SoftmaxLoss::computeLossAndGrad(np::ArrayCpu<float>&x,np::ArrayCpu<int>&y){
        auto shifted_x=x-x.max(1);
         
        auto exp_x=np::exp(shifted_x);
        
        auto denom=exp_x.sum(1);
        
        auto scores = exp_x/denom;
        scores=scores+1e-8;

        auto loss=(-np::log(scores.get(np::arange<int>(x.rows),y))).sum()/x.rows;
        scores.set(np::arange(x.rows), y, NP_OP_SUB, 1);

        auto dx=scores/x.rows;
        return {loss,dx};


    }

