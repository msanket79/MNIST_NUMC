#ifndef NP_ARRAY_CPU_HPP
#define NP_ARRAY_CPU_HPP
#include "customKernels.hpp"
#include<cstring>
#include<cmath>
#include<iostream>
#include<vector>
#include<type_traits>
#include<omp.h>
#include<cblas.h>
#include<algorithm>


namespace np 

{
#define NP_OP_ADD 1
#define NP_OP_SUB 2
#define NP_OP_MUL 3
#define NP_OP_DIV 4
#define NP_OP_LESS_THAN 5
#define NP_OP_LESS_THAN_EQ 6
#define NP_OP_GREATER_THAN 7
#define NP_OP_GREATER_THAN_EQ 8
#define NP_OP_EQEQ 9
#define NP_OP_NOT_EQ 10
#define NP_OP_MINIMUM 11
#define NP_OP_MAXIMUM 12

#define NP_OP_EQ 13

#define NP_REDUCE_SUM 14
#define NP_REDUCE_MIN 15
#define NP_REDUCE_MAX 16
#define NP_REDUCE_ARGMIN 17
#define NP_REDUCE_ARGMAX 18

#define NP_F_EXP 19
#define NP_F_LOG 20
#define NP_F_SQAURE 21
#define NP_F_SQRT 22
#define NP_F_POW 23

    template<typename TP>
    class ArrayCpu  
        {

        public:
            TP* mat;
            int* ref_count;
            int rows, cols;

            ArrayCpu(const int rows = 1,const int cols = 1);

            ArrayCpu(const int rows,const int cols,const TP val);

            ArrayCpu(const std::vector<TP>&A);
    
            ArrayCpu(const std::vector<std::vector<TP>>&A);

            ArrayCpu(const TP* A, const int rows, const int cols);


            ArrayCpu(const ArrayCpu& other);

            void operator=(const ArrayCpu& other);

            void operator=(const TP val);

             unsigned int size() const;

            void print() const;

            void print(int _rows,int _cols) const;            


            ArrayCpu<TP> copy() const;

            void reshape(const int rows,const int cols);

            ArrayCpu<TP> T() const;

            ~ArrayCpu();

            ArrayCpu<TP> at(const int idx) ;

        
            ArrayCpu<TP> at(const int i,const  int j);

            TP& operator()(const int i,const int j);

            TP get(const int i,const int j) const;
            TP get(const int idx) const;
            ArrayCpu<TP> get(const std::vector<int>&idxs) const;
            ArrayCpu<TP> get(const ArrayCpu<int>&idxs) const;
            ArrayCpu<TP> get(const ArrayCpu<int>&r,const ArrayCpu<int>&c) const;
            ArrayCpu<TP> get(std::vector<int>&r,std::vector<int>&c) const;

            void set(const int idx,const char op,const TP operand=0);
            void set(const int r,const int c,const char op,const TP operand=0);
            void set(const ArrayCpu<int>&idxs,const char op,const TP operand=0);
            void set(const std::vector<int>&idxs,const char op,const TP operand=0);
            void set(const ArrayCpu<int>&r,const ArrayCpu<int>&c,const char op,const TP operand=0);
            void set(const std::vector<int>&r,const std::vector<int>&c,const char op,const TP operand=0);

            void set(const std::vector<int>&idxs,const char op,const std::vector<TP>& operand);
            void set(const ArrayCpu<int>&idxs,const char op,const const ArrayCpu<TP> &operand);
            void set(const ArrayCpu<int>&r,const ArrayCpu<int>&c,const char op,const ArrayCpu<TP> &operand);
            void set(const std::vector<int>&r,const std::vector<int>&c,const char op,const std::vector<TP>&operand);

            ArrayCpu<TP> dot(const ArrayCpu<TP>& B)const;

            ArrayCpu<TP> Tdot(const ArrayCpu<TP>& B)const ;

            ArrayCpu<TP> dotT(const ArrayCpu<TP>& B)const ;

            

            void operator<<(const TP* A)const;

            ArrayCpu<TP> operator+(const ArrayCpu<TP>& B)const ;
            ArrayCpu<TP> operator+(const TP b)const;

            ArrayCpu<TP> operator-(const ArrayCpu<TP>& B)const ;
            ArrayCpu<TP> operator-(const TP b)const ;

            ArrayCpu<TP> operator-() const;


            ArrayCpu<TP> operator*(const ArrayCpu<TP>& B)const ;
            ArrayCpu<TP> operator*(const TP b)const ;

            ArrayCpu<TP> operator/(const ArrayCpu<TP>& B)const ;
            ArrayCpu<TP> operator/(const TP b)const ;

            
            ArrayCpu<TP> operator>(const ArrayCpu& B)const ;
            ArrayCpu<TP> operator>(const TP b)const ;

            ArrayCpu<TP> operator<(const ArrayCpu& B)const;
            ArrayCpu<TP> operator<(const TP b)const;

            ArrayCpu<TP> operator>=(const ArrayCpu& B)const;
            ArrayCpu<TP> operator>=(const TP b)const ;

            ArrayCpu<TP> operator<=(const ArrayCpu& B)const;
            ArrayCpu<TP> operator<=(const TP b)const;

            ArrayCpu<TP> operator==(const ArrayCpu<TP>&B) const;
            ArrayCpu<TP> operator==(const TP b) const;

            ArrayCpu<TP> operator!=(const ArrayCpu<TP>&B) const;
            ArrayCpu<TP> operator!=(const TP b) const;
            
            ArrayCpu<TP> sum(const int axis = -1) const;

            ArrayCpu<TP> max(const int axis = -1) const;

            ArrayCpu<TP> min(const int axis = -1)const ;

            ArrayCpu<int> argmin(const int axis = -1) const ;

            ArrayCpu<int> argmax(const int axis = -1) const;

            ArrayCpu<TP> sort(const int axis = -1) const;
            


    };

    template<typename TP>
    ArrayCpu<TP>::ArrayCpu(const int rows,const int cols) {
        this->rows = rows;
        this->cols = cols;
        this->mat = (TP*)malloc(sizeof(TP) * rows * cols);
        this->ref_count = new int;
        *this->ref_count=1;
    }
    template<typename TP>
    ArrayCpu<TP>::ArrayCpu(const TP*A,const int rows,const int cols) {
        this->rows = rows;
        this->cols = cols;
        this->mat = (TP*)malloc(sizeof(TP) * rows * cols);
        std::memcpy(this->mat,A,sizeof(TP)*rows*cols);
        this->ref_count = new int;
        *this->ref_count=1;
    }

    template<typename TP>
    ArrayCpu<TP>::ArrayCpu(const int rows,const int cols,const TP val){
        this->rows = rows;
        this->cols = cols;
        this->mat=(TP*)malloc(sizeof(TP) *this->rows *this->cols);
        this->ref_count = new int;
        *this->ref_count=1;
        //memset works fine for only 0 && -1 ,sometimes it works for other values sometime it doesn't 
        //so making my own initialization function
        InitMat<TP>(mat, rows, cols, val);
    }

    template<typename TP>
    ArrayCpu<TP>::ArrayCpu(const std::vector<TP>&A){
        this->rows=1;
        this->cols=A.size();
        this->ref_count=(int*)malloc(sizeof(int));
        *this->ref_count=1;
        this->mat=(TP*)malloc(sizeof(TP) *this->rows *this->cols);
        std::memcpy(this->mat,A.data,sizeof(TP)*this->rows*this->cols);
    }

    template<typename TP>
    ArrayCpu<TP>::ArrayCpu(const std::vector<std::vector<TP>>&A){
        this->rows=A.size();
        this->cols=A[0].size();
        this->ref_count=(int*)malloc(sizeof(int));
        *this->ref_count=1;
        this->mat=(TP*)malloc(sizeof(TP) *this->rows *this->cols);
        std::memcpy(this->mat,A.data,sizeof(TP)*this->rows*this->cols);
    }

    template<typename TP>
    ArrayCpu<TP>::ArrayCpu(const ArrayCpu<TP>& other){
        this->rows = other.rows;
        this->cols = other.cols;
        this->mat = other.mat;
        this->ref_count = other.ref_count;
        ++(*this->ref_count);
    }

    template<typename TP>
    void ArrayCpu<TP>::operator=(const ArrayCpu& other) {
        if (this != &other) {
            this->rows = other.rows;
            this->cols = other.cols;
            --(*this->ref_count);
            if (*this->ref_count == 0) {
                delete[]this->mat;
                this->mat = nullptr;
            }
            this->mat = other.mat;
            this->ref_count = other.ref_count;
            ++(*this->ref_count);
        }
    }

    template<typename TP>
    void ArrayCpu<TP>::operator=(const TP val){
        InitMat<TP>(this->mat,this->rows,this->cols,val);
    }

    template<typename TP>
     unsigned int ArrayCpu<TP>::size() const{
        return this->rows*this->cols;
    }

    template<typename TP>
    void ArrayCpu<TP>::print() const{
        
        std::cout << "dimensions: " << this->rows << " X " << this->cols << "\n";
        for (int i = 0; i <this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                std::cout << this->mat[i * this->cols + j] << " ";
            }
            std::cout << "\n";
        }
    }

    template<typename TP>
    void ArrayCpu<TP>::print(int _rows,int _cols) const{
        
        std::cout << "dimensions: " << this->rows << " X " << this->cols << "\n";
        for (int i = 0; i <_rows; i++) {
            for (int j = 0; j < _cols; j++) {
                std::cout << this->mat[i * this->cols + j] << " ";
            }
            std::cout << "\n";
        }
    }

    template<typename TP>
    ArrayCpu<TP> ArrayCpu<TP>::copy() const{
        auto A = ArrayCpu<TP>(this->rows, this->cols);
        std::memcpy(A.mat, this->mat, this->rows *this->cols * sizeof(TP));
        return A;
    }

    template<typename TP>
    void ArrayCpu<TP>::reshape(const int rows,const int cols) {
        if (rows * cols == this->rows * this->cols) {
            this->rows = cols;
            this->cols = rows;
        }
    }
// parallel it
    template<typename TP>
    ArrayCpu<TP> ArrayCpu<TP>::T() const  {
        ArrayCpu<TP> matT = ArrayCpu<TP>(cols, rows);
        transposeRec(0, this->rows - 1, 0, this->cols - 1, this->mat, matT.mat, this->rows, this->cols);
        return matT;

    }

    template<typename TP>
    ArrayCpu<TP>::~ArrayCpu() {
        --(*this->ref_count);
        if (*this->ref_count == 0) {
            delete[]mat;
            mat = nullptr;
        }
    }

    template<typename TP>
    ArrayCpu<TP> ArrayCpu<TP>::at(const int idx) {
        auto A=ArrayCpu<TP>(1,1);
        A.mat=this->mat+idx;
        A.ref_count=this->ref_count;
        ++(*this->ref_count);
    }



    template<typename TP>
    ArrayCpu<TP> ArrayCpu<TP>::at(const int i,const int j)  {
        return this->at(i*this->cols+j);
    }

    template<typename TP>
    TP& ArrayCpu<TP>::operator()(const int i,const int j) {
        return this->mat[i * this->cols + j];
    }

    template<typename TP>
    TP ArrayCpu<TP>::get(const int i,const int j) const{
        return this.get(i*this->cols+j);
    }

    template<typename TP>
    TP ArrayCpu<TP>::get(const int idx) const{
        return this->mat[idx];
    }   

    template<typename TP>
    ArrayCpu<TP> ArrayCpu<TP>::get(const std::vector<int>&idxs) const{
        return this->get(ArrayCpu<int>(idxs));
    }

    template<typename TP>
    ArrayCpu<TP> ArrayCpu<TP>::get(const ArrayCpu<int>&idxs) const{
        int vecDim=std::max(idxs.rows,idxs.cols);
        auto B=ArrayCpu<TP>(1,vecDim);
        kernelGetMat<TP>(this->mat,B.mat,idxs.mat,vecDim);
        return B;
    };
    
    template<typename TP>
    ArrayCpu<TP> ArrayCpu<TP>::get(const ArrayCpu<int>&r,const ArrayCpu<int>&c) const{
        int VecDim=std::max(r.rows,r.cols);
        auto B=ArrayCpu<TP>(1, VecDim);
     
            
        kernelGetMat<TP>(this->mat,B.mat,this->cols,r.mat,c.mat,VecDim);
        return B;
    
    }
    template<typename TP>
    ArrayCpu<TP> ArrayCpu<TP>::get(std::vector<int>&r,std::vector<int>&c)const {
        return this->get(ArrayCpu<int>(r),ArrayCpu<int>(c));
    }

    
    template<typename TP>
    void ArrayCpu<TP>::set(const int idx,const char op,const TP operand){
        auto idxs=ArrayCpu<int>(1,1,idx);
        int sz=1;
        switch(op){
            case 1:
                kernelSetMat<TP,1>(this->mat,operand,idxs.mat,sz);
                break;
            case 2:
                kernelSetMat<TP,2>(this->mat,operand,idxs.mat,sz);
                break;
            case 3:
                kernelSetMat<TP,3>(this->mat,operand,idxs.mat,sz);
                break;
            case 4:
                kernelSetMat<TP,4>(this->mat,operand,idxs.mat,sz);
                break;
            case 5:
                kernelSetMat<TP,5>(this->mat,operand,idxs.mat,sz);
                break;
            case 6:
                kernelSetMat<TP,6>(this->mat,operand,idxs.mat,sz);
                break;
            case 7:
                kernelSetMat<TP,7>(this->mat,operand,idxs.mat,sz);
                break;
            case 8:
                kernelSetMat<TP,8>(this->mat,operand,idxs.mat,sz);
                break;
            case 9:
                kernelSetMat<TP,9>(this->mat,operand,idxs.mat,sz);
                break;
            case 10:
                kernelSetMat<TP,10>(this->mat,operand,idxs.mat,sz);
                break;
            case 11:
                kernelSetMat<TP,11>(this->mat,operand,idxs.mat,sz);
                break;
            case 12:
                kernelSetMat<TP,12>(this->mat,operand,idxs.mat,sz);
                break;
            case 13:
                kernelSetMat<TP,13>(this->mat,operand,idxs.mat,sz);
                break;
            default:
                std::cout<<"wrong operand passed in set\n";

        }
    }

    template<typename TP>
    void ArrayCpu<TP>::set(const int r,const int c,const char op,const TP operand){
        this->set(r*this->cols+c,op,operand);
    }

    template<typename TP>
    void ArrayCpu<TP>::set(const std::vector<int>&idxs,const char op,const TP operand){
        this->set(ArrayCpu<int>(idxs),op,operand);
    }

    template<typename TP>
    void ArrayCpu<TP>::set(const ArrayCpu<int>&idxs,const char op,const const TP operand){
        int sz=std::max(idxs.rows,idxs.cols);
        switch(op){
            case 1:
                kernelSetMat<TP,1>(this->mat,operand,idxs.mat,sz);
                break;
            case 2:
                kernelSetMat<TP,2>(this->mat,operand,idxs.mat,sz);
                break;
            case 3:
                kernelSetMat<TP,3>(this->mat,operand,idxs.mat,sz);
                break;
            case 4:
                kernelSetMat<TP,4>(this->mat,operand,idxs.mat,sz);
                break;
            case 5:
                kernelSetMat<TP,5>(this->mat,operand,idxs.mat,sz);
                break;
            case 6:
                kernelSetMat<TP,6>(this->mat,operand,idxs.mat,sz);
                break;
            case 7:
                kernelSetMat<TP,7>(this->mat,operand,idxs.mat,sz);
                break;
            case 8:
                kernelSetMat<TP,8>(this->mat,operand,idxs.mat,sz);
                break;
            case 9:
                kernelSetMat<TP,9>(this->mat,operand,idxs.mat,sz);
                break;
            case 10:
                kernelSetMat<TP,10>(this->mat,operand,idxs.mat,sz);
                break;
            case 11:
                kernelSetMat<TP,11>(this->mat,operand,idxs.mat,sz);
                break;
            case 12:
                kernelSetMat<TP,12>(this->mat,operand,idxs.mat,sz);
                break;
            case 13:
                kernelSetMat<TP,13>(this->mat,operand,idxs.mat,sz);
                break;
            default:
                std::cout<<"wrong operand passed in set\n";

        }
    }

    
    template<typename TP>
    void ArrayCpu<TP>::set(const ArrayCpu<int>&idxs,const char op,const const ArrayCpu<TP>&operand){
        int sz=std::max(idxs.rows,idxs.cols);
        switch(op){
            case 1:
                kernelSetMat<TP,1>(this->mat,operand.mat,idxs.mat,sz);
                break;
            case 2:
                kernelSetMat<TP,2>(this->mat,operand.mat,idxs.mat,sz);
                break;
            case 3:
                kernelSetMat<TP,3>(this->mat,operand.mat,idxs.mat,sz);
                break;
            case 4:
                kernelSetMat<TP,4>(this->mat,operand.mat,idxs.mat,sz);
                break;
            case 5:
                kernelSetMat<TP,5>(this->mat,operand.mat,idxs.mat,sz);
                break;
            case 6:
                kernelSetMat<TP,6>(this->mat,operand.mat,idxs.mat,sz);
                break;
            case 7:
                kernelSetMat<TP,7>(this->mat,operand.mat,idxs.mat,sz);
                break;
            case 8:
                kernelSetMat<TP,8>(this->mat,operand.mat,idxs.mat,sz);
                break;
            case 9:
                kernelSetMat<TP,9>(this->mat,operand.mat,idxs.mat,sz);
                break;
            case 10:
                kernelSetMat<TP,10>(this->mat,operand.mat,idxs.mat,sz);
                break;
            case 11:
                kernelSetMat<TP,11>(this->mat,operand.mat,idxs.mat,sz);
                break;
            case 12:
                kernelSetMat<TP,12>(this->mat,operand.mat,idxs.mat,sz);
                break;
            case 13:
                kernelSetMat<TP,13>(this->mat,operand.mat,idxs.mat,sz);
                break;
            default:
                std::cout<<"wrong operand passed in set\n";

        }
    }

    template<typename TP>
    void ArrayCpu<TP>::set(const std::vector<int>&idxs,const char op,const const std::vector<TP>&operand){
        this->set(ArrayCpu<int>(idxs),op,ArrayCpu<TP>(operand));
    }
    template<typename TP>
    void ArrayCpu<TP>::set(const ArrayCpu<int>&r,const ArrayCpu<int>&c,const char op,const ArrayCpu<TP> &operand){
        int sz=std::max(r.rows,r.cols);
        switch(op){
            case 1:
                kernelSetMat<TP,1>(this->mat,operand.mat,r.mat,c.mat,this->cols,sz);
                break;
            case 2:
                kernelSetMat<TP,2>(this->mat,operand.mat,r.mat,c.mat,this->cols,sz);
                break;
            case 3:
                kernelSetMat<TP,3>(this->mat,operand.mat,r.mat,c.mat,this->cols,sz);
                break;
            case 4:
                kernelSetMat<TP,4>(this->mat,operand.mat,r.mat,c.mat,this->cols,sz);
                break;
            case 5:
                kernelSetMat<TP,5>(this->mat,operand.mat,r.mat,c.mat,this->cols,sz);
                break;
            case 6:
                kernelSetMat<TP,6>(this->mat,operand.mat,r.mat,c.mat,this->cols,sz);
                break;
            case 7:
                kernelSetMat<TP,7>(this->mat,operand.mat,r.mat,c.mat,this->cols,sz);
                break;
            case 8:
                kernelSetMat<TP,8>(this->mat,operand.mat,r.mat,c.mat,this->cols,sz);
                break;
            case 9:
                kernelSetMat<TP,9>(this->mat,operand.mat,r.mat,c.mat,this->cols,sz);
                break;
            case 10:
                kernelSetMat<TP,10>(this->mat,operand.mat,r.mat,c.mat,this->cols,sz);
                break;
            case 11:
                kernelSetMat<TP,11>(this->mat,operand.mat,r.mat,c.mat,this->cols,sz);
                break;
            case 12:
                kernelSetMat<TP,12>(this->mat,operand.mat,r.mat,c.mat,this->cols,sz);
                break;
            case 13:
                kernelSetMat<TP,13>(this->mat,operand.mat,r.mat,c.mat,this->cols,sz);
                break;
            default:
                std::cout<<"wrong operand passed in set\n";

        }
    }

    template<typename TP>
    void ArrayCpu<TP>::set(const std::vector<int>&r,const std::vector<int>&c,const char op,const std::vector<TP>&operand){
        this->set(ArrayCpu<int>(r),ArrayCpu<int>(c),op,ArrayCpu<TP>(operand));
    }

    template<typename TP>
    void ArrayCpu<TP>::set(const ArrayCpu<int>&r,const ArrayCpu<int>&c,const char op,const TP operand){
        int sz=std::max(r.rows,r.cols);
        switch(op){
            case 1:
                kernelSetMat<TP,1>(this->mat,operand,r.mat,c.mat,this->cols,sz);
                break;
            case 2:
                kernelSetMat<TP,2>(this->mat,operand,r.mat,c.mat,this->cols,sz);
                break;
            case 3:
                kernelSetMat<TP,3>(this->mat,operand,r.mat,c.mat,this->cols,sz);
                break;
            case 4:
                kernelSetMat<TP,4>(this->mat,operand,r.mat,c.mat,this->cols,sz);
                break;
            case 5:
                kernelSetMat<TP,5>(this->mat,operand,r.mat,c.mat,this->cols,sz);
                break;
            case 6:
                kernelSetMat<TP,6>(this->mat,operand,r.mat,c.mat,this->cols,sz);
                break;
            case 7:
                kernelSetMat<TP,7>(this->mat,operand,r.mat,c.mat,this->cols,sz);
                break;
            case 8:
                kernelSetMat<TP,8>(this->mat,operand,r.mat,c.mat,this->cols,sz);
                break;
            case 9:
                kernelSetMat<TP,9>(this->mat,operand,r.mat,c.mat,this->cols,sz);
                break;
            case 10:
                kernelSetMat<TP,10>(this->mat,operand,r.mat,c.mat,this->cols,sz);
                break;
            case 11:
                kernelSetMat<TP,11>(this->mat,operand,r.mat,c.mat,this->cols,sz);
                break;
            case 12:
                kernelSetMat<TP,12>(this->mat,operand,r.mat,c.mat,this->cols,sz);
                break;
            case 13:
                kernelSetMat<TP,13>(this->mat,operand,r.mat,c.mat,this->cols,sz);
                break;
            default:
                std::cout<<"wrong operand passed in set\n";

        }
    }

    template<typename TP>
    void ArrayCpu<TP>::set(const std::vector<int>&r,const std::vector<int>&c,const char op,const TP operand){
        this->set(ArrayCpu<int>(r),ArrayCpu<int>(c),op,operand);
    }


    template<typename TP>
    ArrayCpu<TP> ArrayCpu<TP>::operator-() const {
        auto A = ArrayCpu<TP>(this->rows, this->cols);
    #pragma omp parallel for
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                A(i, j) = -this->mat[i * cols + j];
            }
        }
        return A;
    }


    template<typename TP>
    void ArrayCpu<TP>::operator<<(const TP* A) const{
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j <this->cols; j++) {
                mat[i * cols + j] = A[i * cols + j];
            }
        }
    }

    template<typename TP>
    ArrayCpu<TP> ArrayCpu<TP>::dot(const ArrayCpu<TP>& B)const  {
         if (std::is_integral<TP>::value) {
             float* A = new float[this->rows * this->cols];
             float* Bmat = new float[B.rows * B.cols];
             float* C = new float[this->rows * B.cols];
             IntToFloat(A,this-> mat, this->rows, this->cols);
             IntToFloat(Bmat, B.mat, B.rows, B.cols);

             cblas_sgemm(
                 CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 this->rows, B.cols, this->cols,
                 1.0,
                 A, this->cols,
                 Bmat, B.cols,
                 0.0,
                 C, B.cols
             );
             delete[]A;
             delete[]Bmat;

             auto Ans = ArrayCpu<TP>(this->rows, B.cols);
             FloatToInt(reinterpret_cast<int*>(Ans.mat), C, this->rows, B.cols);
             delete[]C;
             return Ans;


         }
        else if (std::is_same<TP, float>::value) {
             
            auto C = ArrayCpu<TP>(this->rows, B.cols);
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                this->rows, B.cols, this->cols,
                1.0,
                reinterpret_cast<float*>(this->mat), this->cols,
                reinterpret_cast<float*>(B.mat), B.cols,
                0.0,
                reinterpret_cast<float*>(C.mat), B.cols
            );
            return C;
        }
        else if (std::is_same<TP, double>::value) {
            auto C = ArrayCpu<TP>(this->rows, B.cols);
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                this->rows, B.cols, this->cols,
                1.0,
                reinterpret_cast<double*>(this->mat), this->cols,
                reinterpret_cast<double*>(B.mat), B.cols,
                0.0,
                reinterpret_cast<double*>(C.mat), B.cols
            );
        }
        else{
            std::cout << "This type is not supported in dot\n";
        }
        return NULL;
    }

    template<typename TP>
    ArrayCpu<TP> ArrayCpu<TP>::dotT(const ArrayCpu<TP>& B)const {
         if (std::is_integral<TP>::value) {
             float* A = new float[this->rows * this->cols];
             float* Bmat = new float[B.rows * B.cols];
             float* C = new float[this->rows * B.rows];
             IntToFloat(A, mat, this->rows, this->cols);
             IntToFloat(Bmat, B.mat, B.rows, B.cols);
             
             cblas_sgemm(
                CblasRowMajor, CblasNoTrans,CblasTrans,
                this->rows,B.rows, this->cols,
                1.0,
                A, this->cols,
                Bmat,B.cols,
                0.0,
                C, B.rows);
             delete[]A;
             delete[]Bmat;
             
             auto Ans = ArrayCpu<TP>(this->rows, B.rows);
             FloatToInt(reinterpret_cast<int*>(Ans.mat), C, this->rows, B.rows);
             delete[]C;
             return Ans;
         }
        else if (std::is_same<TP, float>::value) {
            auto C = ArrayCpu<TP>(this->rows, B.rows);

            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                this->rows, B.rows, this->cols,
                1.0,
                reinterpret_cast<float*>(this->mat), this->cols,
                reinterpret_cast<float*>(B.mat), B.cols,
                0.0,
                reinterpret_cast<float*>(C.mat), B.rows
            );
            return C;
        }
        else if (std::is_same<TP, double>::value) {
            auto C = ArrayCpu<TP>(this->rows, B.rows);
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                this->rows, B.rows, this->cols,
                1.0,
                reinterpret_cast<double*>(this->mat), this->cols,
                reinterpret_cast<double*>(B.mat), B.cols,
                0.0,
                reinterpret_cast<double*>(C.mat), B.rows
            ); 
            return C;

        }
        else std::cout<<"This type is not supported in dotT\n";
        return NULL;
    }

    template<typename TP>
    ArrayCpu<TP> ArrayCpu<TP>::Tdot(const ArrayCpu<TP>& B)const {
        if (std::is_integral<TP>::value) {
            float* A = new float[this->rows * this->cols];
            float* Bmat = new float[B.rows * B.cols];
            float* C = new float[this->cols * B.cols];
            IntToFloat(A, mat, this->rows, this->cols);
            IntToFloat(Bmat, B.mat, B.rows, B.cols);

            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                this->cols, B.cols,this->rows,
                1.0,
                A, this->cols,
                Bmat, B.cols,
                0.0,
                C, B.cols
            );


            delete[]A;
            delete[]Bmat;
            auto Ans = ArrayCpu<TP>(this->cols, B.cols);
            FloatToInt(reinterpret_cast<int*>(Ans.mat), C, this->cols, B.cols);
            delete[]C;
            return Ans;

        }
            
            if (std::is_same<TP, float>::value) {
                auto C = ArrayCpu<TP>(this->cols, B.cols);

                cblas_sgemm(
                    CblasRowMajor, CblasTrans, CblasNoTrans,
                    this->cols, B.cols, this->rows,
                    1.0,
                    reinterpret_cast<float*>(this->mat), this->cols,
                    reinterpret_cast<float*>(B.mat), B.cols,
                    0.0,
                    reinterpret_cast<float*>(C.mat), B.cols
                );

                return C;
            }
            else if (std::is_same<TP, double>::value) {
                auto C = ArrayCpu<TP>(this->cols, B.cols);

                cblas_dgemm(
                    CblasRowMajor, CblasTrans, CblasNoTrans,
                    this->cols, B.cols, this->rows,
                    1.0,
                    reinterpret_cast<double*>(this->mat), this->cols,
                    reinterpret_cast<double*>(B.mat), B.cols,
                    0.0,
                    reinterpret_cast<double*>(C.mat), B.cols
                );

                return C;

            }
            else std::cout << "This type is not supported in Tdot\n";
            return NULL;
        }


        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator+(const ArrayCpu<TP>&B) const {
            //when A is scalaer

            if (this->rows == 1 && this->cols == 1) {

                auto C = ArrayCpu<TP>(B.rows, B.cols);
                kernelMatAddScalar<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
                return C;

            }
            else if (B.rows == 1 && B.cols == 1) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatAddScalar<TP>(this->mat, B.mat[0], C.mat, this->rows, this->cols);
                return C;
            }
            else if (this->rows == B.rows && this->cols == B.cols) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatAddMat<TP>(this->mat, B.mat, C.mat, rows, cols);
                return C;
            }
            else if (this->rows == 1 || this->cols == 1) {

                int vecDim = this->rows > this->cols ? this->rows : this->cols;
                //when the second matrix is square matrix then exception
                if (vecDim == B.rows && vecDim == B.cols) {

                }
                else if (vecDim == B.rows) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelMatAddVecAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
                else if (vecDim == B.cols) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelMatAddVecAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
            }
            else if (B.rows == 1 || B.cols == 1) {

                int vecDim = B.rows > B.cols ? B.rows : B.cols;
                
                if (vecDim == this->rows && vecDim == this->cols) {

                }
                else if (vecDim == this->rows) {

                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatAddVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
                else if (vecDim == this->cols) {
                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatAddVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
            }

        }


        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator+(const TP b) const {
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatAddScalar<TP>(this->mat, b, C.mat, this->rows, this->cols);
            return C;

        }

        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator-(const ArrayCpu<TP>&B) const {
            //when A is scalaer

            if (this->rows == 1 && this->cols == 1) {

                auto C = ArrayCpu<TP>(B.rows, B.cols);
                kernelScalarSubMat<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
                return C;

            }
            else if (B.rows == 1 && B.cols == 1) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatSubScalar<TP>(this->mat, B.mat[0], C.mat, this->rows, this->cols);
                return C;
            }
            else if (this->rows == B.rows && this->cols == B.cols) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatSubMat<TP>(this->mat, B.mat, C.mat, rows, cols);
                return C;
            }
            else if (this->rows == 1 || this->cols == 1) {

                int vecDim = this->rows > this->cols ? this->rows : this->cols;
                //when the second matrix is square matrix then exception
                if (vecDim == B.rows && vecDim == B.cols) {
                    // ??
                }
                else if (vecDim == B.rows) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelVecSubMatAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
                else if (vecDim == B.cols) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelVecSubMatAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
            }
            else if (B.rows == 1 || B.cols == 1) {

                int vecDim = B.rows > B.cols ? B.rows : B.cols;
                
                if (vecDim == this->rows && vecDim == this->cols) {

                }
                else if (vecDim == this->rows) {

                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatSubVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
                else if (vecDim == this->cols) {
                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatSubVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
            }

        }


        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator-(const TP b)const {
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatSubScalar<TP>(this->mat, b, C.mat, this->rows, this->cols);
            return C;

        }
        //--

        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator*(const ArrayCpu<TP>&B)const {
            //when A is scalaer

            if (this->rows == 1 && this->cols == 1) {

                auto C = ArrayCpu<TP>(B.rows, B.cols);
                kernelMatMulScalar<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
                return C;

            }
            else if (B.rows == 1 && B.cols == 1) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatMulScalar<TP>(this->mat, B.mat[0], C.mat, this->rows, this->cols);
                return C;
            }
            else if (this->rows == B.rows && this->cols == B.cols) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatMulMat<TP>(this->mat, B.mat, C.mat, rows, cols);
                return C;
            }
            else if (this->rows == 1 || this->cols == 1) {

                int vecDim = this->rows > this->cols ? this->rows : this->cols;
                //when the second matrix is square matrix then exception
                if (vecDim == B.rows && vecDim == B.cols) {

                }
                else if (vecDim == B.rows) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelMatMulVecAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
                else if (vecDim == B.cols) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelMatMulVecAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
            }
            else if (B.rows == 1 || B.cols == 1) {

                int vecDim = B.rows > B.cols ? B.rows : B.cols;
                
                if (vecDim == this->rows && vecDim == this->cols) {

                }
                else if (vecDim == this->rows) {

                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatMulVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
                else if (vecDim == this->cols) {
                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatMulVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
            }

        }


        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator*(const TP b)const {
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatMulScalar<TP>(this->mat, b, C.mat, this->rows, this->cols);
            return C;

        }

        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator/(const ArrayCpu<TP>&B)const {
            //when A is scalaer
            

            if (this->rows == 1 && this->cols == 1) {

                auto C = ArrayCpu<TP>(B.rows, B.cols);
                kernelScalarDivMat<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
                return C;

            }
            else if (B.rows == 1 && B.cols == 1) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatDivScalar<TP>(this->mat, B.mat[0], C.mat, this->rows, this->cols);
                return C;
            }
            else if (this->rows == B.rows && this->cols == B.cols) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatDivMat<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                return C;
            }
            else if (this->rows == 1 || this->cols == 1) {

                int vecDim = this->rows > this->cols ? this->rows : this->cols;
                //when the second matrix is square matrix then exception
                if (vecDim == B.rows && vecDim == B.cols) {

                }
                else if (vecDim == B.rows) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelVecDivMatAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
                else if (vecDim == B.cols) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelVecDivMatAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
            }
            else if (B.rows == 1 || B.cols == 1) {
              
                int vecDim = B.rows > B.cols ? B.rows : B.cols;
                
                if (vecDim == this->rows && vecDim == this->cols) {

                }
                else if (vecDim == this->rows) {
                 
                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatDivVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
                else if (vecDim == this->cols) {
                  
                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatDivVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
            }

        }


        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator/(const TP b)const {
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatDivScalar<TP>(this->mat, b, C.mat, this->rows, this->cols);
            return C;

        }

        //----

        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator>(const ArrayCpu<TP>&B)const {
            //when A is scalaer

            if (this->rows == 1 && this->cols == 1) {

                auto C = ArrayCpu<TP>(B.rows, B.cols);
                kernelScalarGreaterThanMat<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
                return C;

            }
            else if (B.rows == 1 && B.cols == 1) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatGreaterThanScalar<TP>(this->mat, B.mat[0], C.mat, this->rows, this->cols);
                return C;
            }
            else if (this->rows == B.rows && this->cols == B.cols) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatGreaterThanMat<TP>(this->mat, B.mat, C.mat, rows, cols);
                return C;
            }
            else if (this->rows == 1 || this->cols == 1) {

                int vecDim = this->rows > this->cols ? this->rows : this->cols;
                //when the second matrix is square matrix then exception
                if (vecDim == B.rows && vecDim == B.cols) {

                }
                else if (vecDim == B.rows) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelVecGreaterThanMatAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
                else if (vecDim == B.cols) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelVecGreaterThanMatAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
            }
            else if (B.rows == 1 || B.cols == 1) {

                int vecDim = B.rows > B.cols ? B.rows : B.cols;
        
                if (vecDim == this->rows && vecDim == this->cols) {

                }
                else if (vecDim == this->rows) {

                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatGreaterThanVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
                else if (vecDim == this->cols) {
                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatGreaterThanVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
            }

        }


        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator>(const TP b) const{
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatGreaterThanScalar<TP>(this->mat, b, C.mat, this->rows, this->cols);
            return C;

        }

        //-----------

        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator<(const ArrayCpu<TP>&B)const {
            //when A is scalaer

            if (this->rows == 1 && this->cols == 1) {

                auto C = ArrayCpu<TP>(B.rows, B.cols);
                kernelScalarLessThanMat<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
                return C;

            }
            else if (B.rows == 1 && B.cols == 1) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatLessThanScalar<TP>(this->mat, B.mat[0], C.mat, this->rows, this->cols);
                return C;
            }
            else if (this->rows == B.rows && this->cols == B.cols) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatLessThanMat<TP>(this->mat, B.mat, C.mat, rows, cols);
                return C;
            }
            else if (this->rows == 1 || this->cols == 1) {

                int vecDim = this->rows > this->cols ? this->rows : this->cols;
                //when the second matrix is square matrix then exception
                if (vecDim == B.rows && vecDim == B.cols) {

                }
                else if (vecDim == B.rows) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelVecLessThanMatAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
                else if (vecDim == B.cols) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelVecLessThanMatAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
            }
            else if (B.rows == 1 || B.cols == 1) {

                int vecDim = B.rows > B.cols ? B.rows : B.cols;
      
                if (vecDim == this->rows && vecDim == this->cols) {

                }
                else if (vecDim == this->rows) {

                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatLessThanVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
                else if (vecDim == this->cols) {
                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatLessThanVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
            }

        }


        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator<(const TP b) const{
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatLessThanScalar<TP>(this->mat, b, C.mat, this->rows, this->cols);
            return C;

        }

        //-----------------------

        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator!=(const ArrayCpu<TP>&B) const {
            //when A is scalaer

            if (this->rows == 1 && this->cols == 1) {

                auto C = ArrayCpu<TP>(B.rows, B.cols);
                kernelMatNotEqualScalar<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
                return C;

            }
            else if (B.rows == 1 && B.cols == 1) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatNotEqualScalar<TP>(this->mat, B.mat[0], C.mat, this->rows, this->cols);
                return C;
            }
            else if (this->rows == B.rows && this->cols == B.cols) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatNotEqualMat<TP>(this->mat, B.mat, C.mat, rows, cols);
                return C;
            }
            else if (this->rows == 1 || this->cols == 1) {

                int vecDim = this->rows > this->cols ? this->rows : this->cols;
                //when the second matrix is square matrix then exception
                if (vecDim == B.rows && vecDim == B.cols) {

                }
                else if (vecDim == B.rows) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelMatNotEqualVecAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
                else if (vecDim == B.cols) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelMatNotEqualVecAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
            }
            else if (B.rows == 1 || B.cols == 1) {

                int vecDim = B.rows > B.cols ? B.rows : B.cols;
           
                if (vecDim == this->rows && vecDim == this->cols) {

                }
                else if (vecDim == this->rows) {

                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatNotEqualVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
                else if (vecDim == this->cols) {
                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatNotEqualVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
            }

        }


        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator!=(const TP b)const {
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatNotEqualScalar<TP>(this->mat, b, C.mat, this->rows, this->cols);
            return C;

        }

        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator==(const ArrayCpu<TP>&B) const {
            //when A is scalaer

            if (this->rows == 1 && this->cols == 1) {

                auto C = ArrayCpu<TP>(B.rows, B.cols);
                kernelMatEqualEqualScalar<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
                return C;

            }
            else if (B.rows == 1 && B.cols == 1) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatEqualEqualScalar<TP>(this->mat, B.mat[0], C.mat, this->rows, this->cols);
                return C;
            }
            else if (this->rows == B.rows && this->cols == B.cols) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatEqualEqualMat<TP>(this->mat, B.mat, C.mat, rows, cols);
                return C;
            }
            else if (this->rows == 1 || this->cols == 1) {

                int vecDim = this->rows > this->cols ? this->rows : this->cols;
                //when the second matrix is square matrix then exception
                if (vecDim == B.rows && vecDim == B.cols) {

                }
                else if (vecDim == B.rows) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelMatEqualEqualVecAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
                else if (vecDim == B.cols) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelMatEqualEqualVecAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
            }
            else if (B.rows == 1 || B.cols == 1) {

                int vecDim = B.rows > B.cols ? B.rows : B.cols;
        
                if (vecDim == this->rows && vecDim == this->cols) {

                }
                else if (vecDim == this->rows) {

                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatEqualEqualVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
                else if (vecDim == this->cols) {
                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatEqualEqualVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
            }

        }

        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator==(const TP b)const {
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatEqualEqualScalar<TP>(this->mat, b, C.mat, this->rows, this->cols);
            return C;

        }

        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator>=(const ArrayCpu<TP>&B)const {
            //when A is scalaer

            if (this->rows == 1 && this->cols == 1) {

                auto C = ArrayCpu<TP>(B.rows, B.cols);
                kernelScalarGreaterThanEqualMat<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
                return C;

            }
            else if (B.rows == 1 && B.cols == 1) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatGreaterThanEqualScalar<TP>(this->mat, B.mat[0], C.mat, this->rows, this->cols);
                return C;
            }
            else if (this->rows == B.rows && this->cols == B.cols) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatGreaterThanEqualMat<TP>(this->mat, B.mat, C.mat, rows, cols);
                return C;
            }
            else if (this->rows == 1 || this->cols == 1) {

                int vecDim = this->rows > this->cols ? this->rows : this->cols;
                //when the second matrix is square matrix then exception
                if (vecDim == B.rows && vecDim == B.cols) {

                }
                else if (vecDim == B.rows) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelVecGreaterThanEqualMatAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
                else if (vecDim == B.cols) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelVecGreaterThanEqualMatAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
            }
            else if (B.rows == 1 || B.cols == 1) {

                int vecDim = B.rows > B.cols ? B.rows : B.cols;
               
                if (vecDim == this->rows && vecDim == this->cols) {

                }
                else if (vecDim == this->rows) {

                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatGreaterThanEqualVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
                else if (vecDim == this->cols) {
                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatGreaterThanEqualVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
            }

        }


        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator>=(const TP b)const  {
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatGreaterThanEqualScalar<TP>(this->mat, b, C.mat, this->rows, this->cols);
            return C;

        }



        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator<=(const ArrayCpu<TP>&B)const  {
            //when A is scalaer

            if (this->rows == 1 && this->cols == 1) {

                auto C = ArrayCpu<TP>(B.rows, B.cols);
                kernelScalarLessThanEqualMat<TP>(B.mat, this->mat[0], C.mat, B.rows, B.cols);
                return C;

            }
            else if (B.rows == 1 && B.cols == 1) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatLessThanEqualScalar<TP>(this->mat, B.mat[0], C.mat, this->rows, this->cols);
                return C;
            }
            else if (this->rows == B.rows && this->cols == B.cols) {

                auto C = ArrayCpu<TP>(this->rows, this->cols);
                kernelMatLessThanEqualMat<TP>(this->mat, B.mat, C.mat, rows, cols);
                return C;
            }
            else if (this->rows == 1 || this->cols == 1) {

                int vecDim = this->rows > this->cols ? this->rows : this->cols;
                //when the second matrix is square matrix then exception
                if (vecDim == B.rows && vecDim == B.cols) {

                }
                else if (vecDim == B.rows) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelVecLessThanEqualMatAlongCols<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
                else if (vecDim == B.cols) {
                    auto C = ArrayCpu<TP>(B.rows, B.cols);
                    kernelVecLessThanEqualMatAlongRows<TP>(B.mat, this->mat, C.mat, B.rows, B.cols);
                    return C;
                }
            }
            else if (B.rows == 1 || B.cols == 1) {

                int vecDim = B.rows > B.cols ? B.rows : B.cols;
                
                if (vecDim == this->rows && vecDim == this->cols) {

                }
                else if (vecDim == this->rows) {

                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatLessThanEqualVecAlongCols<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
                else if (vecDim == this->cols) {
                    auto C = ArrayCpu<TP>(this->rows, this->cols);
                    kernelMatLessThanEqualVecAlongRows<TP>(this->mat, B.mat, C.mat, this->rows, this->cols);
                    return C;
                }
            }

        }


        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::operator<=(const TP b)const {
            auto C = ArrayCpu<TP>(this->rows, this->cols);
            kernelMatLessThanEqualScalar<TP>(this->mat, b, C.mat, this->rows, this->cols);
            return C;

        }



        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::sum(const int axis) const {
            if (axis == 0) {
                const ArrayCpu<TP> A = this->T();
                auto B = A.sum(1);
                std::swap(B.rows, B.cols);
                return B;
            }
            else if (axis == 1) {
                auto SumRow = ArrayCpu<TP>(rows, 1, 0);
#pragma omp parallel for
                for (int i = 0; i < rows; i++) {
                    TP ans = 0;
                    for (int j = 0; j < cols; j++) {
                        ans += this->mat[i * this->cols + j];
                    }
                    SumRow(i, 0) = ans;
                }
                return SumRow;
            }
            else {
                TP ans = 0;
#pragma omp parallel for reduction(+:ans)
                for (int i = 0; i < this->rows * this->cols; i++) {
                    ans += this->mat[i];
                }

                return ArrayCpu<TP>(1, 1, ans);

            }
        }


        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::max(const int axis) const {
            if (axis == 0) {
                const ArrayCpu<TP> A = this->T();
                auto B = A.max(1);
                std::swap(B.rows, B.cols);
                return B;
            }
            else if (axis == 1) {
                auto SumRow = ArrayCpu<TP>(rows, 1, 0);
#pragma omp parallel for
                for (int i = 0; i < rows; i++) {
                    TP ans = this->mat[i * this->cols];
                    for (int j = 1; j < cols; j++) {
                        if (this->mat[i * this->cols + j] > ans) ans = this->mat[i * this->cols + j];
                    }
                    SumRow(i, 0) = ans;
                }
                return SumRow;
            }
            else {
                TP ans = this->mat[0];
#pragma omp parallel for reduction(max:ans)
                for (int i = 0; i < this->rows * this->cols; i++) {
                    if (this->mat[i] > ans) {
                        ans = this->mat[i];
                    }
                }
                return ArrayCpu<TP>(1, 1, ans);

            }
        }


        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::min(const int axis) const {
            if (axis == 0) {
                const ArrayCpu<TP> A = this->T();
                auto B = A.min(1);
                std::swap(B.rows, B.cols);
                return B;
            }
            else if (axis == 1) {
                auto SumRow = ArrayCpu<TP>(rows, 1, 0);
#pragma omp parallel for
                for (int i = 0; i < rows; i++) {
                    TP ans = this->mat[i * this->cols];
                    for (int j = 1; j < cols; j++) {
                        if (this->mat[i * this->cols + j] < ans) ans = this->mat[i * this->cols + j];
                    }
                    SumRow(i, 0) = ans;
                }
                return SumRow;
            }
            else {
                TP ans = this->mat[0];
#pragma omp parallel for reduction(min:ans)
                for (int i = 0; i < this->rows * this->cols; i++) {
                    if (this->mat[i] < ans) {
                        ans = this->mat[i];
                    }
                }
                return ArrayCpu<TP>(1, 1, ans);

            }
        }


        template<typename TP>
        ArrayCpu<int> ArrayCpu<TP>::argmin(const int axis) const {
            if (axis == 0) {
                const ArrayCpu<TP> A = this->T();
                auto B = A.argmin(1);
                std::swap(B.rows, B.cols);
                return B;
            }
            else if (axis == 1) {
                auto SumRow = ArrayCpu<int>(rows, 1, 0);
#pragma omp parallel for
                for (int i = 0; i < rows; i++) {
                    int ind = 0;
                    for (int j = 1; j < cols; j++) {
                        if (this->mat[i * this->cols + j] < this->mat[i * this->cols + ind]) ind = j;
                    }
                    SumRow(i, 0) = ind;
                }
                return SumRow;
            }
            else {
                int ind = 0;
                for (int j = 1; j < this->rows * this->cols; j++) {
                    if (this->mat[j] < this->mat[ind]) ind = j;
                }
                int x = ind / this->cols;
                int y = ind % this->cols;
                auto ans = ArrayCpu<int>(1, 2);
                ans.mat[0] = x;
                ans.mat[1] = y;
                return ans;

            }
        }


        template<typename TP>
        ArrayCpu<int> ArrayCpu<TP>::argmax(const int axis) const {
            if (axis == 0) {
                const ArrayCpu<TP> A = this->T();
                auto B = A.argmax(1);
                std::swap(B.rows, B.cols);
                return B;
            }
            else if (axis == 1) {
                auto SumRow = ArrayCpu<int>(this->rows, 1, 0);
#pragma omp parallel for
                for (int i = 0; i < this->rows; i++) {
                    int ind = 0;
                    int rowadd = i * this->cols;
                    for (int j = 1; j < this->cols; j++) {
                        if (this->mat[rowadd + j] > this->mat[rowadd + ind])  ind = j;
                    }
                    SumRow(i, 0) = ind;
                }
                return SumRow;
            }

            else {
                int ind = 0;
                for (int j = 1; j < this->rows * this->cols; j++) {
                    if (this->mat[j] > this->mat[ind]) ind = j;
                }
                int x = ind / this->cols;
                int y = ind % this->cols;
                auto ans = ArrayCpu<int>(1, 2);
                ans.mat[0] = x;
                ans.mat[1] = y;
                return ans;

            }
        }


        template<typename TP>
        std::ostream& operator<<(std::ostream & out, const ArrayCpu<TP>&Arr) {
            out << "dimensions: " << Arr.rows << " X " << Arr.cols << "\n";
            for (int i = 0; i < Arr.rows; i++) {
                for (int j = 0; j < Arr.cols; j++) {
                    out << Arr.mat[i * Arr.cols + j] << " ";
                }
                out << "\n";
            }
            return out;
        }

        template<typename TP>
        ArrayCpu<TP> ArrayCpu<TP>::sort(const int axis) const {
            if (axis == 1) {
                auto A = this->T();
                auto B = A.sort(0);
                return B.T();
            }
            else if (axis == 0) {
                auto A = this->copy();
#pragma omp parallel for
                for (int i = 0; i < A.rows; i++) {
                    kernelParallelMergeSortWrapper<TP>(A.mat + i * A.cols, 0, A.cols - 1);
                }
                return A;
            }
            else {
                auto A = this->copy();
                kernelParallelMergeSortWrapper<TP>(A.mat, 0, A.rows * A.cols - 1);
                return A;
            }

        }
  }


#endif // !