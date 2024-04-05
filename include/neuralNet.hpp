#ifndef NEURALNET_HPP
#define NEURALNET_HPP

#include<layers/affineLayer.hpp>
#include<layers/dropoutLayer.hpp>
#include<layers/reluLayer.hpp>

// loss function
#include<lossFunctions/softmax.hpp>

#include<optimizers/adam.hpp>

#include<numC/npArrayCpu.hpp>

#include<string>
#include<vector>

class NeuralNet{
    public:
    std::vector<AffineLayer> affine_Layers;
    std::vector<ReluLayer> relu_Layers;
    std::vector<DropoutLayer>dropout_layers;
    std::vector<AdamOptimizer> adam_configs;

    std::string mode;

    NeuralNet(const float reg=0.0,float p_keep=1.0);
    NeuralNet(NeuralNet&N);
    void operator=(const NeuralNet&N);

    void train();
    void test();


    np::ArrayCpu<float> forward(const np::ArrayCpu<float> &X);


    std::pair<np::ArrayCpu<float>, np::ArrayCpu<float>> forward(const np::ArrayCpu<float> &X, const np::ArrayCpu<int> &y);

    np::ArrayCpu<float> operator()(const np::ArrayCpu<float> &X);
    std::pair<np::ArrayCpu<float>, np::ArrayCpu<float>> operator()(const np::ArrayCpu<float> &X, const np::ArrayCpu<int> &y);

    np::ArrayCpu<float> backward(np::ArrayCpu<float> &dout);


    void adamStep();

}


#endif