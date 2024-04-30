#include <layers/affineLayer.hpp>
#include <layers/reluLayer.hpp>
#include <layers/dropoutLayer.hpp>

#include <lossFunctions/softmax.hpp>

#include<optimizers/adam.hpp>

#include <neuralNet.hpp>

#include<numC/npArrayCpu.hpp>

#include <iostream>
#include <string>
#include <vector>
    NeuralNet::NeuralNet(const float p_keep){
        // first layer
        this->l1=AffineLayer(784,2048);
        this->l2=DropoutLayer(p_keep);
        this->l3=ReluLayer();
        this->l4=AffineLayer(2048,10);
        this->l1_w=AdamOptimizer(0.001f, 0.9f, 0.999f, 1e-8f);
        this->l1_b=AdamOptimizer(0.001f, 0.9f, 0.999f, 1e-8f);
        this->l4_w=AdamOptimizer(0.001f, 0.9f, 0.999f, 1e-8f);
        this->l4_b=AdamOptimizer(0.001f, 0.9f, 0.999f, 1e-8f);
        this->mode="eval";
    }
    NeuralNet::NeuralNet(const  NeuralNet&N){
        this->l1=N.l1;
        this->l2=N.l2;
        this->l3=N.l3;
        this->l4=N.l4;
        this->l1_w=N.l1_w;
        this->l1_w=N.l1_w;
        this->l4_w=N.l4_w;
        this->l4_b=N.l4_b;
        this->mode=N.mode;
    }
    void NeuralNet::operator=(const NeuralNet&N){
        this->l1=N.l1;
        this->l2=N.l2;
        this->l3=N.l3;
        this->l4=N.l4;
        this->l1_w=N.l1_w;
        this->l1_w=N.l1_w;
        this->l4_w=N.l4_w;
        this->l4_b=N.l4_b;
        this->mode=N.mode;
    }

    void NeuralNet::train(){
        this->mode="train";
    }
    void NeuralNet::eval(){
        this->mode="eval";
    }

    // foward pass (used in eval mode to only return the output)
    np::ArrayCpu<float> NeuralNet::forward( np::ArrayCpu<float>&X){
        if(this->mode=="train"){
            std::cout<<"Labels not passed for training\n";
            return NULL;
        }
        auto out=X;
        out=l1.forward(out,this->mode);
        out=l2.forward(out,this->mode);
        out=l3.forward(out,this->mode);
        out=l4.forward(out,this->mode);
        return out;
    }
    // used in train mode returns out and loss
    std::pair<np::ArrayCpu<float>,np::ArrayCpu<float>> NeuralNet::forward( np::ArrayCpu<float>&X, np::ArrayCpu<int>&Y){
        auto out=X;
        out=l1.forward(out,this->mode);
        out=l2.forward(out,this->mode);
        out=l3.forward(out,this->mode);
        out=l4.forward(out,this->mode);

        if(mode=="eval"){
           
            return {out,SoftmaxLoss::computeLoss(out,Y)};
        }
         //std::cout<<"forward train mode\n";
        // if training mode is there now output is calculate and we will calculate dL of the softmax loss function and pass it to backward prop 
        auto lossNgrad=SoftmaxLoss::computeLossAndGrad(out,Y);

        this->backward(lossNgrad[1]);
        
        return {out,lossNgrad[0]};
    
    }

    //now overloading for forward pass

    // foward pass (used in eval mode to only return the output)
    np::ArrayCpu<float> NeuralNet::operator()( np::ArrayCpu<float>&X){
        return this->forward(X);
    }
    // used in train mode returns out and loss
    std::pair<np::ArrayCpu<float>,np::ArrayCpu<float>> NeuralNet::operator()( np::ArrayCpu<float>&X, np::ArrayCpu<int>&Y){
        return this->forward(X,Y);
    }

    np::ArrayCpu<float> NeuralNet::backward(np::ArrayCpu<float>&dOut){
        dOut=this->l4.backward(dOut);
        dOut=this->l3.backward(dOut);
        dOut=this->l2.backward(dOut);
        dOut=this->l1.backward(dOut);
        return dOut;

    }


    void NeuralNet::adamStep(){
        this->l4_w.step(l4.w,l4.dw);
        this->l4_b.step(l4.b,l4.db);
        this->l1_w.step(l1.w,l1.dw);
        this->l1_b.step(l1.b,l1.db);

    }