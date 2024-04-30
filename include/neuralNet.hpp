#ifndef NEURALNET_HPP
#define NEURALNET_HPP
#include<layers/affineLayer.hpp>
#include<layers/dropoutLayer.hpp>
#include<layers/reluLayer.hpp>

#include<lossFunctions/softmax.hpp>
#include<optimizers/adam.hpp>

#include<numC/npArrayCpu.hpp>
#include<string>
#include<vector>

class NeuralNet{
    public:
    // layer1 
    AffineLayer l1;
    // layer 2
    DropoutLayer l2;
    // layer 3
    ReluLayer l3;
    //layer 4
    AffineLayer l4;
    // std::vector<AdamOptimizer> adam_configs;
    AdamOptimizer l1_w;
    AdamOptimizer l1_b;
    AdamOptimizer l4_w;
    AdamOptimizer l4_b;
    // eval | train
    std::string mode;
    NeuralNet(const float p_keep=1.0);
    NeuralNet(const NeuralNet&N);
    void operator=(const NeuralNet&N);
    void train();
    void eval();

    // foward pass (used in eval mode to only return the output)
    np::ArrayCpu<float> forward( np::ArrayCpu<float>&X);
    // used in train mode returns out and loss
    std::pair<np::ArrayCpu<float>,np::ArrayCpu<float>> forward( np::ArrayCpu<float>&X, np::ArrayCpu<int>&Y);

    //now overloading for forward pass

    // foward pass (used in eval mode to only return the output)
    np::ArrayCpu<float> operator()( np::ArrayCpu<float>&X);
    // used in train mode returns out and loss
    std::pair<np::ArrayCpu<float>,np::ArrayCpu<float>> operator()( np::ArrayCpu<float>&X, np::ArrayCpu<int>&Y);

    np::ArrayCpu<float> backward(np::ArrayCpu<float>&dOut);


    void adamStep();



};
#endif