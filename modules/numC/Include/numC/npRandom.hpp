#ifndef NP_RANDOM_HPP
#define NP_RANDOM_HPP
#include "npArrayCpu.hpp"
#include<time.h>
#include<type_traits>
#include<random>
#include<iostream>
#include<chrono>

namespace np {
	class Random {
	public:
		//for uniform distribution
		template<typename TP>
		static ArrayCpu<TP> rand(int rows = 1, int cols = 1, int lo = 0, int hi = 1, unsigned long long seed = (std::chrono::high_resolution_clock::now().time_since_epoch().count()));
		template<typename TP>
		static ArrayCpu<TP> rand(int rows, int cols, unsigned long long seed);

		//for normal distribution

		template<typename TP>
		static ArrayCpu<TP> randn(int rows=1, int cols=1, unsigned long long seed = (std::chrono::high_resolution_clock::now().time_since_epoch().count()));

	};

	template<typename TP>
	static ArrayCpu<TP> Random::rand(int rows, int cols, int lo, int hi, unsigned long long seed) {
		auto A = ArrayCpu<TP>(rows, cols);
		
		if (std::is_integral<TP>::value) {
			#pragma omp parallel
			{
				std::uniform_int_distribution<int> distribution(lo, hi);
				std::default_random_engine generator(seed + omp_get_thread_num());
			#pragma omp for
				for (int i = 0; i < rows * cols; i++) {
					A.mat[i] = distribution(generator);
				}
			}
			return A;
		
		}
		else {
#pragma omp parallel
			{
				std::uniform_real_distribution<double> distribution(lo, hi);
				std::default_random_engine generator(seed + omp_get_thread_num());
#pragma omp for
				for (int i = 0; i < rows * cols; i++) {
					A.mat[i] = distribution(generator);
				}
			}
			return A;

		}

		}
	
	
	template<typename TP>
	static ArrayCpu<TP> Random::rand(int rows, int cols, unsigned long long seed) {
		auto A = ArrayCpu<TP>(rows, cols);

		if (std::is_integral<TP>::value) {
#pragma omp parallel
			{
				std::uniform_int_distribution<int> distribution(0, 1);
				std::default_random_engine generator(seed + omp_get_thread_num());
#pragma omp for
				for (int i = 0; i < rows * cols; i++) {
					A.mat[i] = distribution(generator);
				}
			}
			return A;
			

		}
		else {
#pragma omp parallel
			{
				std::uniform_real_distribution<double> distribution(0, 1);
				std::default_random_engine generator(seed + omp_get_thread_num());
#pragma omp for
				for (int i = 0; i < rows * cols; i++) {
					A.mat[i] = distribution(generator);
				}
			}
			return A;

		}
	}

	template<typename TP>
	static ArrayCpu<TP> Random::randn(int rows, int cols, unsigned long long seed) {
		auto A = ArrayCpu<TP>(rows, cols);
		#pragma omp parallel
		{
		std::normal_distribution<float>distribution(0, 1);
		std::default_random_engine generator(seed+omp_get_thread_num());
		#pragma omp for
		for(int i=0;i<rows*cols;i++){
			A.mat[i] = distribution(generator);
		}
		}
		
		return A;
	}





}




#endif