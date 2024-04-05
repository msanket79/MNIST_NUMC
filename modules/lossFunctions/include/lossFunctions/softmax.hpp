#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP
#include "numC/npArrayCpu.hpp"
#include "numC/npFunctions.hpp"
#include<vector>
class SoftmaxLoss{
    public:
    static std::vector<np::ArrayCpu<float>> computeLossAndGrad(np::ArrayCpu<float>&x, np::ArrayCpu<int>&y);
    static np::ArrayCpu<float> computeLoss(np::ArrayCpu<float>&x, np::ArrayCpu<int>&y);
};
#endif
