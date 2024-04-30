#ifndef CUSTOM_KERNELS_HPP
#define CUSTOM_KERNELS_HPP
#include<omp.h>
#include<algorithm>
#include<iostream>

//adding matrices with broadcasting
template<typename TP>
inline void kernelMatAddMat(const TP* A,const TP* B,TP* C,const int rows,const int cols);

template<typename TP>
inline void kernelMatAddScalar(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatAddVecAlongRows(const TP* A, const TP* B,  TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatAddVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

//subtracting matrices with broadcasting
template<typename TP>
inline void kernelMatSubMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatSubScalar(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatSubVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatSubVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelScalarSubMat(const TP*A,const TP b,TP*C,const int rows,const int cols);

template<typename TP>
inline void kernelVecSubMatAlongRows(const TP*A,const TP* B,TP*C,const int rows,const int cols);

template<typename TP>
inline void kernelVecSubMatAlongCols(const TP*A,const TP* B,TP*C,const int rows,const int cols);

//multiplying
template<typename TP>
inline void kernelMatMulMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatMulScalar(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatMulVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatMulVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

//dividing
template<typename TP>
inline void kernelMatDivMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatDivScalar(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatDivVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatDivVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelScalarDivMat(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelVecDivMatAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelVecDivMatAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

//GreaterThan
template<typename TP>
inline void kernelMatGreaterThanMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatGreaterThanScalar(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatGreaterThanVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatGreaterThanVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);


template<typename TP>
inline void kernelScalarGreaterThanMat(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelVecGreaterThanMatAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelVecGreaterThanMatAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

//GreaterThanEqual
template<typename TP>
inline void kernelMatGreaterThanEqualMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatGreaterThanEqualScalar(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatGreaterThanEqualVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatGreaterThanEqualVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelScalarGreaterThanEqualMat(const TP* A, const TP b, TP* C, const int rows, const int cols);
template<typename TP>
inline void kernelVecGreaterThanEqualMatAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);
template<typename TP>
inline void kernelVecGreaterThanEqualMatAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

//LessThan
template<typename TP>
inline void kernelMatLessThanMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatLessThanScalar(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatLessThanVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatLessThanVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);


template<typename TP>
inline void kernelScalarLessThanMat(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelVecLessThanMatAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelVecLessThanMatAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);
//EqualEqual
template<typename TP>
inline void kernelMatEqualEqualMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatEqualEqualScalar(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatEqualEqualVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatEqualEqualVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);
//NotEqual
template<typename TP>
inline void kernelMatNotEqualMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatNotEqualScalar(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatNotEqualVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatNotEqualVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

//LessThanEqual
template<typename TP>
inline void kernelMatLessThanEqualMat(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatLessThanEqualScalar(const TP* A, const TP b, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatLessThanEqualVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelMatLessThanEqualVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);

template<typename TP>
inline void kernelScalarLessThanEqualMat(const TP* A, const TP b, TP* C, const int rows, const int cols);
template<typename TP>
inline void kernelVecLessThanEqualMatAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols);
template<typename TP>
inline void kernelVecLessThanEqualMatAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols);




//misc
template<typename TP>
void FloatToInt(int* B, TP* mat, int rows, int cols);

template<typename TP>
void IntToFloat(float* B, TP* mat, int rows, int cols);

template<typename TP>
void InitMat(TP* mat, int rows, int cols, TP val);

//maximum-------------------
template<typename TP>
inline void kernelMatMaximumScalar(const TP* A,const TP b, TP*C,const int rows,const int cols);
template<typename TP>
inline void kernelMatMaximumMat(const TP* A,const TP* B, TP*C,const int rows,const int cols);
template<typename TP>
inline void kernelMatMaximumVecAlongCols(const TP* A, const TP* B, TP*C,const int rows,const int cols);
template<typename TP>
inline void kernelMatMaximumVecAlongRows(const TP* A,const TP* B, TP*C,const int rows,const int cols);

//minimum-------------------
template<typename TP>
inline void kernelMatMinimumScalar(const TP* A,const TP b, TP*C,const int rows,const int cols);
template<typename TP>
inline void kernelMatMinimumMat(const TP* A,const TP* B, TP*C,const int rows,const int cols);
template<typename TP>
inline void kernelMatMinimumVecAlongCols(const TP* A, const TP* B, TP*C,const int rows,const int cols);
template<typename TP>
inline void kernelMatMinimumVecAlongRows(const TP* A,const TP* B, TP*C,const int rows,const int cols);


// kernelGet-------------
template<typename TP>
void kernelGetMat( TP*A, TP*res,const int*idxs,const int sz);

template<typename TP>
void kernelGetMat( TP*A, TP*res,const int cols,const int*r,const int*c,const int sz);

//kernelSet------------------

template<typename TP,int op>
void kernelSetMat(TP*mat,const TP operand,const int*idxs,const int sz);
template<typename TP>
void kernelSetMat(TP*mat,const TP* operand,const int*idxs,const int sz);
template<typename TP>
void kernelSetMat(TP*mat,const TP operand,const int*r,const int*c,const int cols,const int sz);
template<typename TP>
void kernelSetMat(TP*mat,const TP* operand,const int*r,const int*c,const int cols,const int sz);

//////-----------------MergeSort Kernels--------------------------
template<typename TP>
inline void kerneLMergeSerial(TP* a_start, TP* a_end, TP* b_start, TP* b_end, TP* C);
template<typename TP>
inline int findSplitPoint(const TP* A, const int l, const int h, const TP& x);
template<typename TP>
inline void kernelParallelMerge(TP* A, int l1, int h1, int l2, int h2, TP* B, int l3, int threshold);
template<typename TP>
inline void kernelParallelMergeWrapper(TP* A, int l, int m, int h, int threshold = 32768);
template<typename TP>
inline void insertionSort(TP* A, int size);
template<typename TP>
inline void kernelParallelMergeSort(TP* A, int l, int h);
template<typename TP>
inline void kernelParallelMergeSortWrapper(TP* A, int l, int h);

//--------------------------transpose-------------------------------
template<typename TP>
inline void transposeRec(int rb, int re, int cb, int ce, TP* A, TP* B, int m, int n);
//function definitions
template<typename TP>
inline void kernelMatAddScalar(const TP* A, const TP b,  TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] + b;
		}
	}
}

template<typename TP>
inline void kernelMatAddMat(const TP* A,const TP* B,TP* C,const int rows,const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
		}
	}
}

template<typename TP>
inline void kernelMatAddVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i*cols+j]=A[i * cols + j] + B[j];
		}
	}
}

template<typename TP>
inline void kernelMatAddVecAlongCols(const TP* A, const TP* B,  TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] + a;
		}
	}
}

///-------------------------------------------------------
template<typename TP>
inline void kernelMatSubScalar(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] - b;
		}
	}
}

template<typename TP>
inline void kernelScalarSubMat(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = b- A[i * cols + j];
		}
	}
}

template<typename TP>
inline void kernelMatSubMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] - B[i * cols + j];
		}
	}
}

template<typename TP>
inline void kernelMatSubVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] - B[j];
		}
	}
}
template<typename TP>
inline void kernelVecSubMatAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = B[j]-A[i * cols + j];
		}
	}
}

template<typename TP>
inline void kernelVecSubMatAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = a- A[i * cols + j] ;
		}
	}
}

template<typename TP>
inline void kernelMatSubVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] - a;
		}
	}
}

//--------------------------------------------------------
template<typename TP>
inline void kernelMatNotEqualScalar(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] != b?1:0);
		}
	}
}

template<typename TP>
inline void kernelMatNotEqualMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] != B[i * cols + j]?1:0);
		}
	}
}

template<typename TP>
inline void kernelMatNotEqualVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] !=B[j]?1:0);
		}
	}
}

template<typename TP>
inline void kernelMatNotEqualVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] != a?1:0);
		}
	}
}
//--------------------------------------------------------
//--------------------------------------------------------
template<typename TP>
inline void kernelMatEqualEqualScalar(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] == b?1:0);
		}
	}
}

template<typename TP>
inline void kernelMatEqualEqualMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] == B[i * cols + j]?1:0);
		}
	}
}

template<typename TP>
inline void kernelMatEqualEqualVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] ==B[j]?1:0);
		}
	}
}

template<typename TP>
inline void kernelMatEqualEqualVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] == a?1:0);
		}
	}
}
//--------------------------------------------------------
template<typename TP>
inline void kernelMatMulScalar(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] * b;
		}
	}
}

template<typename TP>
inline void kernelMatMulMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] * B[i * cols + j];
		}
	}
}

template<typename TP>
inline void kernelMatMulVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] * B[j];
		}
	}
}

template<typename TP>
inline void kernelMatMulVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] * a;
		}
	}
}

//--------------------------------------------------------

template<typename TP>
inline void kernelMatDivScalar(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] / b;
		}
	}
}
template<typename TP>
inline void kernelScalarDivMat(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = b /A[i * cols + j];
		}
	}
}

template<typename TP>
inline void kernelMatDivMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] / B[i * cols + j];
		}
	}
}

template<typename TP>
inline void kernelMatDivVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] / B[j];
		}
	}
}

template<typename TP>
inline void kernelMatDivVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {

#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = A[i * cols + j] / a;
		}
	}
}
template<typename TP>
inline void kernelVecDivMatAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = B[j]/A[i * cols + j];
		}
	}
}

template<typename TP>
inline void kernelVecDivMatAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] =a/ A[i * cols + j];
		}
	}
}

//-------------------------------------------------------

template<typename TP>
inline void kernelMatGreaterThanScalar(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] > b ? 1:0);
		}
	}
}
template<typename TP>
inline void kernelScalarGreaterThanMat(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] < b ? 1:0);
		}
	}
}

template<typename TP>
inline void kernelMatGreaterThanMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] > B[i * cols + j] ?1:0);
		}
	}
}

template<typename TP>
inline void kernelMatGreaterThanVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] > B[j]?1:0);
		}
	}
}

template<typename TP>
inline void kernelMatGreaterThanVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] > a ?1:0);
		}
	}
}
template<typename TP>
inline void kernelVecGreaterThanMatAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] < B[j]?1:0);
		}
	}
}

template<typename TP>
inline void kernelVecGreaterThanMatAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] < a ?1:0);
		}
	}
}

//-------------------------------------------------------

template<typename TP>
inline void kernelMatGreaterThanEqualScalar(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] >= b ? 1 : 0);
		}
	}
}
template<typename TP>
inline void kernelScalarGreaterThanEqualMat(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] <= b ? 1 : 0);
		}
	}
}




template<typename TP>
inline void kernelMatGreaterThanEqualMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] >= B[i * cols + j] ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatGreaterThanEqualVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] >= B[j] ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatGreaterThanEqualVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] >= a ? 1 : 0);
		}
	}
}
template<typename TP>
inline void kernelVecGreaterThanEqualMatAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] <= B[j] ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelVecGreaterThanEqualMatAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] <= a ? 1 : 0);
		}
	}
}



//--------------------------------------------------

template<typename TP>
inline void kernelMatLessThanScalar(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] < b ? 1 : 0);
		}
	}
}
template<typename TP>
inline void kernelScalarLessThanMat(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] > b ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatLessThanMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] < B[i * cols + j] ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatLessThanVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] < B[j] ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatLessThanVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] < a ? 1 : 0);
		}
	}
}
template<typename TP>
inline void kernelVecLessThanMatAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] > B[j] ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelVecLessThanMatAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] > a ? 1 : 0);
		}
	}
}


//-----------------------------------------------------------------------------

template<typename TP>
inline void kernelMatLessThanEqualScalar(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] <= b ? 1 : 0);
		}
	}
}
template<typename TP>
inline void kernelScalarLessThanEqualMat(const TP* A, const TP b, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] <= b ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatLessThanEqualMat(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] <= B[i * cols + j] ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatLessThanEqualVecAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] <= B[j] ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelMatLessThanEqualVecAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] <= a ? 1 : 0);
		}
	}
}


template<typename TP>
inline void kernelVecLessThanEqualMatAlongRows(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] >= B[j] ? 1 : 0);
		}
	}
}

template<typename TP>
inline void kernelVecLessThanEqualMatAlongCols(const TP* A, const TP* B, TP* C, const int rows, const int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		TP a = B[i];
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = (A[i * cols + j] >= a ? 1 : 0);
		}
	}
}


template<typename TP>
void InitMat(TP* mat, int rows, int cols, TP val) {
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			mat[i * cols + j] = val;
		}
	}
}
template<typename TP>
void IntToFloat(float* B, TP* mat, int rows, int cols) {

#pragma omp parallel for
	for (int i = 0; i < rows * cols; i++) {
		B[i] = static_cast<float>(mat[i]);
	}

}
template<typename TP>
void FloatToInt(int* B, TP* mat, int rows, int cols) {
#pragma omp parallel for
	for (int i = 0; i < rows * cols; i++) {
		B[i] = static_cast<int>(mat[i]);
	}

}

//---------------------------------------
template<typename TP>
inline void kernelMatMaximumScalar(const TP* A,const TP b, TP*C, const int rows,const int cols){
	#pragma omp parallel for
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			C[i*cols+j]=A[i*cols+j]>b?A[i*cols+j]:b;
		}
	}
}
template<typename TP>
inline void kernelMatMaximumMat(const TP* A,const TP* B, TP*C,const int rows,const int cols){
	#pragma omp parallel for
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			C[i*cols+j]=A[i*cols+j]>B[i*cols+j]?A[i*cols+j]:B[i*cols+j];
		}
	}
}
template<typename TP>
inline void kernelMatMaximumVecAlongCols(const TP* A,const TP* B, TP*C,const int rows,const int cols){
	#pragma omp parallel for
	for(int i=0;i<rows;i++){
		TP b=B[i];
		for(int j=0;j<cols;j++){
			C[i*cols+j]=A[i*cols+j]>b?A[i*cols+j]:b;
		}
	}
}
template<typename TP>
inline void kernelMatMaximumVecAlongRows(const TP* A,const TP* B, TP*C,const int rows,const int cols){
	#pragma omp parallel for
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			C[i*cols+j]=A[i*cols+j]>B[j]?A[i*cols+j]:B[j];
		}
	}
}
//---------------------------------------
template<typename TP>
inline void kernelMatMinimumScalar(const TP* A,const TP b, TP*C, const int rows,const int cols){
	#pragma omp parallel for
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			C[i*cols+j]=A[i*cols+j]<b?A[i*cols+j]:b;
		}
	}
}
template<typename TP>
inline void kernelMatMinimumMat(const TP* A,const TP* B, TP*C,const int rows,const int cols){
	#pragma omp parallel for
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			C[i*cols+j]=A[i*cols+j]<B[i*cols+j]?A[i*cols+j]:B[i*cols+j];
		}
	}
}
template<typename TP>
inline void kernelMatMinimumVecAlongCols(const TP* A,const TP* B, TP*C,const int rows,const int cols){
	#pragma omp parallel for
	for(int i=0;i<rows;i++){
		TP b=B[i];
		for(int j=0;j<cols;j++){
			C[i*cols+j]=A[i*cols+j]<b?A[i*cols+j]:b;
		}
	}
}
template<typename TP>
inline void kernelMatMinimumVecAlongRows(const TP* A,const TP* B, TP*C,const int rows,const int cols){
	#pragma omp parallel for
	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			C[i*cols+j]=A[i*cols+j]<B[j]?A[i*cols+j]:B[j];
		}
	}
}


//----------------------------------------
template<typename TP>
void kernelGetMat( TP*A, TP*res,const int*idxs,const int sz){
	#pragma omp parallel for
	for(int i=0;i<sz;i++){
		res[i]=A[idxs[i]];
	}
}

template<typename TP>
void kernelGetMat( TP*A, TP*res,const int cols,const int*r,const int*c,const int sz){

	#pragma omp parallel for
	for(int i=0;i<sz;i++){
		
		res[i]=A[r[i]*cols+c[i]];
	}
}
// #define NP_OP_ADD 1
// #define NP_OP_SUB 2
// #define NP_OP_MUL 3
// #define NP_OP_DIV 4
// #define NP_OP_LESS_THAN 5
// #define NP_OP_LESS_THAN_EQ 6
// #define NP_OP_GREATER_THAN 7
// #define NP_OP_GREATER_THAN_EQ 8
// #define NP_OP_EQEQ 9
// #define NP_OP_NOT_EQ 10
// #define NP_OP_MINIMUM 11
// #define NP_OP_MAXIMUM 12
// #define NP_OP_EQ 13
template<typename TP,int OP>
void kernelSetMat(TP*mat,const TP operand,const int*idxs,const int sz){
	for(int i=0;i<sz;i++){
		if constexpr(OP==1){
			mat[idxs[i]]+=operand;
		}
		else if constexpr(OP==2){
			mat[idxs[i]]-=operand;
		}
		else if constexpr(OP==3){
			mat[idxs[i]]*=operand;
		}
		else if constexpr(OP==4){
			mat[idxs[i]]/=operand;
		}
		else if constexpr(OP==5){
			mat[idxs[i]]=mat[idxs[i]]<operand?1:0;
		}
		else if constexpr(OP==6){
			mat[idxs[i]]=mat[idxs[i]]<=operand?1:0;
		}
		else if constexpr(OP==7){
			mat[idxs[i]]=mat[idxs[i]]>operand?1:0;
		}
		else if constexpr(OP==8){
			mat[idxs[i]]=mat[idxs[i]]>=operand?1:0;
		}
		else if constexpr(OP==9){
			mat[idxs[i]]=mat[idxs[i]]==operand?1:0;
		}
		else if constexpr(OP==10){
			mat[idxs[i]]=mat[idxs[i]]!=operand?1:0;
		}	
		else if constexpr(OP==11){
			mat[idxs[i]]=std::min(mat[idxs[i]],operand);
		}
		else if constexpr(OP==12){
			mat[idxs[i]]=std::max(mat[idxs[i]],operand);
		}
		else if constexpr(OP==13){
			mat[idxs[i]]=operand;
		}
		else std::cout<<"error in kernelsel\n";
	}
}

template<typename TP,int OP>
void kernelSetMat(TP*mat,const TP* operand,const int*idxs,const int sz){
	for(int i=0;i<sz;i++){
		if constexpr(OP==1){
			mat[idxs[i]]+=operand[i];
		}
		else if constexpr(OP==2){
			mat[idxs[i]]-=operand[i];
		}
		else if constexpr(OP==3){
			mat[idxs[i]]*=operand[i];
		}
		else if constexpr(OP==4){
			mat[idxs[i]]/=operand[i];
		}
		else if constexpr(OP==5){
			mat[idxs[i]]=mat[idxs[i]]<operand[i]?1:0;
		}
		else if constexpr(OP==6){
			mat[idxs[i]]=mat[idxs[i]]<=operand[i]?1:0;
		}
		else if constexpr(OP==7){
			mat[idxs[i]]=mat[idxs[i]]>operand[i]?1:0;
		}
		else if constexpr(OP==8){
			mat[idxs[i]]=mat[idxs[i]]>=operand[i]?1:0;
		}
		else if constexpr(OP==9){
			mat[idxs[i]]=mat[idxs[i]]==operand[i]?1:0;
		}
		else if constexpr(OP==10){
			mat[idxs[i]]=mat[idxs[i]]!=operand[i]?1:0;
		}	
		else if constexpr(OP==11){
			mat[idxs[i]]=std::min(mat[idxs[i]],operand[i]);
		}
		else if constexpr(OP==12){
			mat[idxs[i]]=std::max(mat[idxs[i]],operand[i]);
		}
		else if constexpr(OP==13){
			mat[idxs[i]]=operand[i];
		}
		else std::cout<<"error in kernelsel\n";
	}
}

template<typename TP,int OP>
void kernelSetMat(TP*mat,const TP operand,const int*r,const int*c,const int cols,const int sz){
	for(int i=0;i<sz;i++){
		if constexpr(OP==1){
			mat[r[i]*cols+c[i]]+=operand;
		}
		else if constexpr(OP==2){
			mat[r[i]*cols+c[i]]-=operand;
		}
		else if constexpr(OP==3){
			mat[r[i]*cols+c[i]]*=operand;
		}
		else if constexpr(OP==4){
			mat[r[i]*cols+c[i]]/=operand;
		}
		else if constexpr(OP==5){
			mat[r[i]*cols+c[i]]=mat[r[i]*cols+c[i]]<operand?1:0;
		}
		else if constexpr(OP==6){
			mat[r[i]*cols+c[i]]=mat[r[i]*cols+c[i]]<=operand?1:0;
		}
		else if constexpr(OP==7){
			mat[r[i]*cols+c[i]]=mat[r[i]*cols+c[i]]>operand?1:0;
		}
		else if constexpr(OP==8){
			mat[r[i]*cols+c[i]]=mat[r[i]*cols+c[i]]>=operand?1:0;
		}
		else if constexpr(OP==9){
			mat[r[i]*cols+c[i]]=mat[r[i]*cols+c[i]]==operand?1:0;
		}
		else if constexpr(OP==10){
			mat[r[i]*cols+c[i]]=mat[r[i]*cols+c[i]]!=operand?1:0;
		}	
		else if constexpr(OP==11){
			mat[r[i]*cols+c[i]]=std::min(mat[r[i]*cols+c[i]],operand);
		}
		else if constexpr(OP==12){
			mat[r[i]*cols+c[i]]=std::max(mat[r[i]*cols+c[i]],operand);
		}
		else if constexpr(OP==13){
			mat[r[i]*cols+c[i]]=operand;
		}
		else std::cout<<"error in kernelsel\n";
	}
}

template<typename TP,int OP>
void kernelSetMat(TP*mat,const TP* operand,const int*r,const int*c,const int cols,const int sz){
	for(int i=0;i<sz;i++){
		if constexpr(OP==1){
			mat[r[i]*cols+c[i]]+=operand[i];
		}
		else if constexpr(OP==2){
			mat[r[i]*cols+c[i]]-=operand[i];
		}
		else if constexpr(OP==3){
			mat[r[i]*cols+c[i]]*=operand[i];
		}
		else if constexpr(OP==4){
			mat[r[i]*cols+c[i]]/=operand[i];
		}
		else if constexpr(OP==5){
			mat[r[i]*cols+c[i]]=mat[r[i]*cols+c[i]]<operand[i]?1:0;
		}
		else if constexpr(OP==6){
			mat[r[i]*cols+c[i]]=mat[r[i]*cols+c[i]]<=operand[i]?1:0;
		}
		else if constexpr(OP==7){
			mat[r[i]*cols+c[i]]=mat[r[i]*cols+c[i]]>operand[i]?1:0;
		}
		else if constexpr(OP==8){
			mat[r[i]*cols+c[i]]=mat[r[i]*cols+c[i]]>=operand[i]?1:0;
		}
		else if constexpr(OP==9){
			mat[r[i]*cols+c[i]]=mat[r[i]*cols+c[i]]==operand[i]?1:0;
		}
		else if constexpr(OP==10){
			mat[r[i]*cols+c[i]]=mat[r[i]*cols+c[i]]!=operand[i]?1:0;
		}	
		else if constexpr(OP==11){
			mat[r[i]*cols+c[i]]=std::min(mat[r[i]*cols+c[i]],operand[i]);
		}
		else if constexpr(OP==12){
			mat[r[i]*cols+c[i]]=std::max(mat[r[i]*cols+c[i]],operand[i]);
		}
		else if constexpr(OP==13){
			mat[r[i]*cols+c[i]]=operand[i];
		}
		else std::cout<<"error in kernelSet\n";
	}
}

//-------------------------Sorting--------------------------------
template<typename TP>
inline void kerneLMergeSerial(TP* a_start, TP* a_end, TP* b_start, TP* b_end, TP* C) {
	if (a_start < a_end && b_start < b_end) {
		while (true) {
			if (*a_start <= *b_start) {
				*C = *a_start;
				a_start++;
				C++;
				if (a_start >= a_end) break;
			}
			else {
				*C = *b_start;
				C++;
				b_start++;
				if (b_start >= b_end) break;
			}
		}
	}
	while (a_start < a_end) {
		*C = *a_start;
		a_start++;
		C++;
	}
	while (b_start < b_end) {
		*C = *b_start;
		C++;
		b_start++;
	}
}


template<typename TP>
inline int findSplitPoint(const TP* A, const int l, const int h, const TP& x) {
	int low = l;
	int high = h + 1;
	while (low < high) {
		long mid = (high - low) / 2 + low;
		if (x <= A[mid]) {
			high = mid;
		}
		else low = mid + 1;
	}
	return low;
}


template<typename TP>
inline void kernelParallelMerge(TP* A, int l1, int h1, int l2, int h2, TP* B, int l3, int threshold) {
	int length_a = h1 - l1 + 1;
	int length_b = h2 - l2 + 1;
	// if both the subarrays are empty 
	if (l1 > h1 && l2 > h2) return;
	// if second subarray bigger than first(making sure first array is bigger)
	if ((length_b) > length_a) {
		std::swap(l1, l2);
		std::swap(h1, h2);
	}
	if (length_a + length_b <= threshold) {
		kerneLMergeSerial<TP>(A + l1, A + h1 + 1, A + l2, A + h2 + 1, B + l3);
		return;
	}
	// find the mid point of the first subarray
	int q1 = (h1 - l1) / 2 + l1; //finding the split point in array 1

	int q2 = findSplitPoint<TP>(A, l2, h2, A[q1]); //finding split point in array 2
	// put where x belongs in output
	int q3 = l3 + (q1 - l1) + (q2 - l2);
	B[q3] = A[q1];
	// now recursively merge A[l1,q1-1] and A[l2,q2-1] into B[0,q3-1]
#pragma omp task
	kernelParallelMerge<TP>(A, l1, q1 - 1, l2, q2 - 1, B, l3, threshold);
#pragma omp task
	kernelParallelMerge<TP>(A, q1 + 1, h1, q2, h2, B, q3 + 1, threshold);
#pragma omp taskwait

}

template<typename TP>
inline void kernelParallelMergeWrapper(TP* A, int l, int m, int h, int threshold) {
	TP* B = (TP*)malloc(sizeof(TP) * (h - l + 1)); //sorted array will be stored into this
	// now calling the parallel merge function
#pragma omp parallel
	{
#pragma omp single
		{
			kernelParallelMerge<TP>(A, l, m, m + 1, h, B, 0, threshold);
			// copy back the merged array from B to the main array A
		}
#pragma omp  for
		for (int i = l; i <= h; i++) {
			A[i] = B[i - l];
		}
	}
	delete[]B;
}

template<typename TP>
inline void insertionSort(TP* A, int size) {

	for (int i = 1; i < size; i++) {
		if (A[i] < A[i - 1]) {
			TP curr_element = A[i];
			A[i] = A[i - 1];
			int j;
			for (j = i - 1; j > 0 && curr_element < A[j - 1]; j--) {
				A[j] = A[j - 1];
			}
			A[j] = curr_element;
		}
	}
}


template<typename TP>
inline void kernelParallelMergeSort(TP* A, int l, int h) {
	if (h < l) {
		return;
	}
	if (h - l <= 128) {
		insertionSort<TP>(A + l, h - l + 1);
		return;
	}
	int mid = (h + l) / 2;
#pragma omp task
	kernelParallelMergeSort<TP>(A, l, mid);
#pragma omp task
	kernelParallelMergeSort<TP>(A, mid + 1, h);
#pragma omp taskwait
	kernelParallelMergeWrapper<TP>(A, l, mid, h);

}
template<typename TP>
inline void kernelParallelMergeSortWrapper(TP* A, int l, int h) {
#pragma omp parallel
#pragma omp single
	{
		kernelParallelMergeSort<TP>(A, l, h);
	}
}

template<typename TP>
inline void transposeRec(int rb, int re, int cb, int ce, TP* A, TP* B, int m, int n) {
	int r = re - rb;
	int c = ce - cb;
	if (r <= 8 && c <= 8) {
		for (int i = rb; i <= re; i++) {
			for (int j = cb; j <= ce; j++) {
				B[j * m + i] = A[i * n + j];
			}
		}
	}
	else if (r >= c) {
		transposeRec(rb, rb + r / 2, cb, ce, A, B, m, n);
		transposeRec(rb + r / 2, re, cb, ce, A, B, m, n);
	}
	else {
		transposeRec(rb, re, cb, cb + c / 2, A, B, m, n);
		transposeRec(rb, re, cb + c / 2, ce, A, B, m, n);
	}
}

#endif // !CUSTOM_KERNELS_HPP

