#include<MNIST/showMNIST.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include<iostream>
#include<string>
typedef unsigned char uchar;
void showMNIST(const uchar*img,const int img_height,const int img_width,const std::string &win_name){
    cv::Mat _img(img_height,img_width,CV_8U,(void*)img);
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(win_name, 200, 200);
    cv::imshow(win_name,_img);
    cv::waitKey(0);
    cv::destroyAllWindows();

}
void showMNIST(const float*img,const int img_height,const int img_width,const std::string &win_name){
    uchar* _img=new uchar[img_height*img_width];
    for(int i=0;i<img_height*img_width;i++){
        _img[i]=img[i];
    }
    showMNIST(_img,img_height,img_width,win_name);
}

