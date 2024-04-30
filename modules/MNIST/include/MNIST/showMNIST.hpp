#ifndef SHOW_MNIST_HPP
#define SHOW_MNIST_HPP


#include<string>
typedef unsigned char uchar;
void showMNIST(const uchar*img,const int img_height,const int img_width,const std::string &win_name);
void showMNIST(const float*img,const int img_height,const int img_width,const std::string &win_name);

#endif
