#ifndef READ_MNIST_HPP
#define READ_MNIST_HPP

#include<string>
typedef unsigned char uchar;
int reverseInt(const int I);
uchar* readMNISTImages(const std::string &path,int &num_images,int& img_size);
uchar* readMNISTLabels(const std::string &path,int &num_labels);

#endif