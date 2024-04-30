#ifndef NP_FUNCTIONS_HPP
#define NP_FUNCTIONS_HPP

#include "npArrayCpu.hpp"
#include "customKernels.hpp"
#include <cstdlib> 
#include<iostream>
#include<chrono>
#include<cstring>

namespace np {
	template<typename TP>
	ArrayCpu<TP> arange(const TP start,const TP stop,const TP step);
	template<typename TP>
	ArrayCpu<TP> arange(const TP stop);

	template <typename TP>
	ArrayCpu<TP> ones(int rows = 1, int cols = 1);

	template<typename TP>
	ArrayCpu<TP> zeros(int rows = 1, int cols = 1);

	template<typename TP>
	ArrayCpu<TP> maximum(ArrayCpu<TP>& A, ArrayCpu<TP>& B);

	template<typename TP>
	ArrayCpu<TP> maximum(ArrayCpu<TP>& A,TP b);

	template<typename TP>
	ArrayCpu<TP> minimum(ArrayCpu<TP>& A, ArrayCpu<TP>& B);

	template<typename TP>
	ArrayCpu<TP> minimum(ArrayCpu<TP>& A,TP b);

	template<typename TP>
	ArrayCpu<TP> exp( ArrayCpu<TP>&A);

	template<typename TP>
	ArrayCpu<TP> log( ArrayCpu<TP>&A);

	template<typename TP>
	ArrayCpu<TP> sqrt(ArrayCpu<TP>& A);

	template<typename TP>
	ArrayCpu<TP> square(ArrayCpu<TP>& A);

	template<typename TP,typename TP1>
	ArrayCpu<TP> pow(ArrayCpu<TP>& A, TP1 n);

	template<typename TP>
	void shuffle(ArrayCpu<TP>& A, unsigned long long seed = (std::chrono::high_resolution_clock::now().time_since_epoch().count()));

	template<typename TP>
	std::vector<np::ArrayCpu<TP>> array_split(ArrayCpu<TP>& A, const int num_parts, int axis);


//function declaration-------------

	template<typename TP>
	np::ArrayCpu<TP> maximum(np::ArrayCpu<TP>&A,np::ArrayCpu<TP>& B) {
		//when A is scalaer

		if (A.rows == 1 and A.cols == 1) {

			auto C = ArrayCpu<TP>(B.rows, B.cols);
			kernelMatMaximumScalar<TP>(B.mat, A.mat[0], C.mat, B.rows, B.cols);
			return C;

		}
		else if (B.rows == 1 and B.cols == 1) {

			auto C = ArrayCpu<TP>(A.rows, A.cols);
			kernelMatMaximumScalar<TP>(A.mat, B(0, 0), C.mat, A.rows, A.cols);
			return C;
		}
		else if (A.rows == B.rows && A.cols == B.cols) {

			auto C = ArrayCpu<TP>(A.rows, A.cols);
			kernelMatMaximumMat<TP>(A.mat, B.mat, C.mat, A.rows, A.cols);
			return C;
		}
		else if (A.rows == 1 || A.cols == 1) {

			int vecDim = A.rows > A.cols ? A.rows : A.cols;
			//when the second matrix is square matrix then exception
			if (vecDim == B.rows && vecDim == B.cols) {

			}
			else if (vecDim == B.rows) {
				auto C = ArrayCpu<TP>(B.rows, B.cols);
				kernelMatMaximumVecAlongCols<TP>(B.mat, A.mat, C.mat, B.rows, B.cols);
				return C;
			}
			else if (vecDim == B.cols) {
				auto C = ArrayCpu<TP>(B.rows, B.cols);
				kernelMatMaximumVecAlongRows<TP>(B.mat, A.mat, C.mat, B.rows, B.cols);
				return C;
			}
		}
		else if (B.rows == 1 || B.cols == 1) {

			int vecDim = B.rows > B.cols ? B.rows : B.cols;
			std::cout << vecDim;
			if (vecDim == A.rows && vecDim == A.cols) {

			}
			else if (vecDim == A.rows) {

				auto C = ArrayCpu<TP>(A.rows, A.cols);
				kernelMatMaximumVecAlongCols<TP>(A.mat, B.mat, C.mat, A.rows, A.cols);
				return C;
			}
			else if (vecDim == A.cols) {
				auto C = ArrayCpu<TP>(A.rows, A.cols);
				kernelMatMaximumVecAlongRows<TP>(A.mat, B.mat, C.mat, A.rows, A.cols);
				return C;
			}
		}

	}

	template<typename TP>
	ArrayCpu<TP> maximum(ArrayCpu<TP>& A,TP b){
		auto C=ArrayCpu<TP>(A.rows,A.cols);
		kernelMatMaximumScalar<TP>(A.mat,b,C.mat,A.rows,A.cols);
		return C;
	}
//--------------------------------------------

	template<typename TP>
	np::ArrayCpu<TP> minimum(np::ArrayCpu<TP>&A,np::ArrayCpu<TP>& B) {
		//when A is scalaer

		if (A.rows == 1 and A.cols == 1) {

			auto C = ArrayCpu<TP>(B.rows, B.cols);
			kernelMatMinimumScalar<TP>(B.mat, A.mat[0], C.mat, B.rows, B.cols);
			return C;

		}
		else if (B.rows == 1 and B.cols == 1) {

			auto C = ArrayCpu<TP>(A.rows, A.cols);
			kernelMatMinimumScalar<TP>(A.mat, B(0, 0), C.mat, A.rows, A.cols);
			return C;
		}
		else if (A.rows == B.rows && A.cols == B.cols) {

			auto C = ArrayCpu<TP>(A.rows, A.cols);
			kernelMatMinimumMat<TP>(A.mat, B.mat, C.mat, A.rows, A.cols);
			return C;
		}
		else if (A.rows == 1 || A.cols == 1) {

			int vecDim = A.rows > A.cols ? A.rows : A.cols;
			//when the second matrix is square matrix then exception
			if (vecDim == B.rows && vecDim == B.cols) {

			}
			else if (vecDim == B.rows) {
				auto C = ArrayCpu<TP>(B.rows, B.cols);
				kernelMatMinimumVecAlongCols<TP>(B.mat, A.mat, C.mat, B.rows, B.cols);
				return C;
			}
			else if (vecDim == B.cols) {
				auto C = ArrayCpu<TP>(B.rows, B.cols);
				kernelMatMinimumVecAlongRows<TP>(B.mat, A.mat, C.mat, B.rows, B.cols);
				return C;
			}
		}
		else if (B.rows == 1 || B.cols == 1) {

			int vecDim = B.rows > B.cols ? B.rows : B.cols;
			std::cout << vecDim;
			if (vecDim == A.rows && vecDim == A.cols) {

			}
			else if (vecDim == A.rows) {

				auto C = ArrayCpu<TP>(A.rows, A.cols);
				kernelMatMinimumVecAlongCols<TP>(A.mat, B.mat, C.mat, A.rows, A.cols);
				return C;
			}
			else if (vecDim == A.cols) {
				auto C = ArrayCpu<TP>(A.rows, A.cols);
				kernelMatMinimumVecAlongRows<TP>(A.mat, B.mat, C.mat, A.rows, A.cols);
				return C;
			}
		}

	}

	template<typename TP>
	ArrayCpu<TP> minimum(ArrayCpu<TP>& A,TP b){
		auto C=ArrayCpu<TP>(A.rows,A.cols);
		kernelMatMinimumScalar<TP>(A,b,C,A.rows,A.rows);
		return C;
	}



//------------------------------------------------
	template<typename TP>
	ArrayCpu<TP> arange(const TP start,const TP stop,const TP step) {
		int cols = ceil((static_cast<double>(stop) - start) / step);
		ArrayCpu<TP> A = ArrayCpu<TP>(1, cols);
		int j = 0;
		for (TP i = start; i < stop; i += step) {
			A(0, j) = i;
			j++;
		}
		return A;


	}

	template<typename TP>
	ArrayCpu<TP> arange(const TP stop) {
		ArrayCpu<TP> A = ArrayCpu<TP>(1, stop);
		for (TP i = 0; i < stop; i += 1) {
			A(0, i) = i;
		}
		return A;


	}
	template <typename TP>
	 ArrayCpu<TP> ones(int rows,int cols) {
		 auto A = ArrayCpu<TP>(rows, cols);
#pragma omp parallel for
		 for (int i = 0; i < rows; i++) {
			 for (int j = 0; j < cols; j++) {
				 A.mat[i * cols + j] = 1;
			 }
		 }
		return A;
	}
	template<typename TP>
	ArrayCpu<TP> zeros(int rows,int cols) {
		auto A = ArrayCpu<TP>(rows, cols,0);
		#pragma omp parallel for
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				A.mat[i * cols + j] = 0;
			}
		}
		return A;
	}
	template<typename TP>
	ArrayCpu<TP> exp(ArrayCpu<TP>&A) {
		auto B = ArrayCpu<TP>(A.rows, A.cols, 0);
		
		#pragma omp parallel for
		for (int i = 0; i < A.rows; i++) {
			for (int j = 0; j < A.cols; j++) {
				B(i, j) = std::exp(static_cast<double>(A(i, j)));
			}
		}
		return B;

	}
	template<typename TP>
	ArrayCpu<TP> log(ArrayCpu<TP>&A) {
		auto B = ArrayCpu<TP>(A.rows, A.cols);
		#pragma omp parallel for
		for (int i = 0; i < A.rows; i++) {
			for (int j = 0; j < A.cols; j++) {
				B(i, j) = std::log(static_cast<double>(A(i, j)));
			}
		}
		return B;
	}

	template<typename TP>
	ArrayCpu<TP> sqrt(ArrayCpu<TP>& A) {
		auto B = ArrayCpu<TP>(A.rows, A.cols);

#pragma omp parallel for
		for (int i = 0; i < A.rows; i++) {
			for (int j = 0; j < A.cols; j++) {
				B(i, j) = std::sqrt(static_cast<double>(A(i, j)));
			}
		}
		return B;

	}

	template<typename TP>
	ArrayCpu<TP> square(ArrayCpu<TP>& A) {
		auto B = ArrayCpu<TP>(A.rows, A.cols);

#pragma omp parallel for
		for (int i = 0; i < A.rows; i++) {
			for (int j = 0; j < A.cols; j++) {
				B(i, j) = std::pow(static_cast<double>(A(i, j)),2.0);
			}
		}
		return B;

	}

	template<typename TP,typename TP1>
	ArrayCpu<TP> pow(ArrayCpu<TP>& A,TP1 n) {
		auto B = ArrayCpu<TP>(A.rows, A.cols, 0);

#pragma omp parallel for
		for (int i = 0; i < A.rows; i++) {
			for (int j = 0; j < A.cols; j++) {
				B(i, j) = std::pow(static_cast<double>(A(i, j)),static_cast<double>(n));
			}
		}
		return B;

	}


	template<typename TP>
	void shuffle(ArrayCpu<TP>& A, unsigned long long seed ) {
		if (A.cols <= 1 && A.rows<=1) return;
		srand(seed);

		for (int i = 0; i < A.cols*A.rows; i++) {
			int idx = rand() % (1 + i);
			std::swap(A.mat[idx], A.mat[i]);
		}
	}

	template<typename TP>
	std::vector<np::ArrayCpu<TP>> array_split(ArrayCpu<TP>& A, const int num_parts, int axis) {
		if (num_parts == 1) return { A };
		std::vector<np::ArrayCpu<TP>> splitted_arrays;
		int size = A.rows;
		int part_size = size / num_parts;
		int remainder = size % num_parts;
		int st_idx = 0;
		for (int i = 0; i < num_parts; i++) {
			int this_part_size = part_size + (i < remainder ? 1 : 0);
			np::ArrayCpu<TP> tmp(A.mat+st_idx,this_part_size, A.cols);
			splitted_arrays.push_back(tmp);
			st_idx += tmp.size();
		}
		return splitted_arrays;
	}
}
#endif