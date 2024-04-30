#include<MNIST/readMNIST.hpp>

#include<iostream>
#include<fstream>
#include<string>

typedef unsigned char uchar;
// we are using reverseint because when MNIST was created many of UNIX systems used bigendian format for storing data hence it is in big endian so we have to reverse to get the exact value
inline int reverseInt(const int i){
    uchar u1,u2,u3,u4;
    u1=i & 255;
    u2=(i>>8)&255;
    u3=(i>>16)&255;
    u4=(i>>24)&255;
    return ((int)u1<<24)+((int)u2 <<16)+((int)u3<<8)+u4;

}

uchar* readMNISTImages(const std::string &path,int &num_images,int& img_size){
    std::ifstream file(path,std::ios::binary);
    
    if(file.is_open()){
        int magic_no=0,rows=0,cols=0;
        file.read((char*)&magic_no,sizeof(magic_no));
        file.read((char*)&num_images,sizeof(num_images));
        file.read((char*)&rows,sizeof(rows));
        file.read((char*)&cols,sizeof(cols));
        magic_no=reverseInt(magic_no);
        // reading the number of images
        num_images=reverseInt(num_images);
        // reading the rows of a image 24*24
        rows=reverseInt(rows);
        // reading the no of columns in a image  24*24
        cols=reverseInt(cols);

        if(magic_no != 2051){
            std::cout<<"Not a valid MNIST file\n";
            return NULL;
        }
        img_size=rows*cols;
        uchar* dataset=new uchar[num_images*img_size];
        file.read(reinterpret_cast<char*>(dataset),num_images*img_size);
        file.close();
        return dataset;
    }
    else{
        std::cout<<"Cannot open file "<<path<<std::endl;

    }
}
uchar* readMNISTLabels(const std::string &path,int &num_labels){
    std::ifstream file(path,std::ios::binary);
    
    if(file.is_open()){
        int magic_no=0,rows=0,cols=0;
        file.read((char*)&magic_no,sizeof(magic_no));
        file.read((char*)&num_labels,sizeof(num_labels));

        magic_no=reverseInt(magic_no);
        num_labels=reverseInt(num_labels);

        if(magic_no != 2049){
            std::cout<<"Not a valid MNIST  label file\n";
            return NULL;
        }
        uchar* dataset=new uchar[num_labels];
        for(int i=0;i<num_labels;i++){
            
        file.read(reinterpret_cast<char*>(&dataset[i]),1);
        }
         file.close();
        return dataset;
    }
    else{
        std::cout<<"Cannot open file "<<path<<std::endl;

    }
}