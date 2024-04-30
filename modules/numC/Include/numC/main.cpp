#include "npArrayCpu.hpp"
#include "npRandom.hpp"
#include "npFunctions.hpp"
#include<iostream>
#include <chrono>

template<typename TP>
void test(TP* A, TP* B, int n) {
	int flag = 1;
#pragma omp parallel
	{
		int flag_private = 1;
#pragma omp for
		for (int i = 0; i < n; i++) {
			if (A[i] != B[i]) {
				flag_private = 0;

			}
		}
#pragma omp critical
		{
			flag = flag & flag_private;
		}

	}
	if (flag == 1) std::cout << "test passed\n";
	else std::cout << "test failed\n";
}
template<typename TP>
void isSorted(TP* A, int n) {
	int flag = 1;
	for(int i=1;i<n;i++){
		if (A[i - 1] > A[i])
		{
			flag = 0;
			break;
		}
	}
	if (flag == 1) std::cout << "Sorted\n";
	else std::cout << "Not Sorted\n";
}
template<typename Func, typename... Args>
double timeit(Func func, Args&&... args) {
	auto start = std::chrono::high_resolution_clock::now();
	func(std::forward<Args>(args)...);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	std::cout << "Time taken: " << duration.count() / 1e6 << "ms" << std::endl;
	return duration.count() / 1e6;
}
int main() {

	int n = 10 ;
	auto A = np::Random::randn<float>(8190, 8190);
	auto start = std::chrono::high_resolution_clock::now();
	auto B = A.sort(0);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	std::cout << "Time taken: " << duration.count() / 1e6 << "ms" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	auto C = A.sort(1);
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	std::cout << "Time taken: " << duration.count() / 1e6 << "ms" << std::endl;

	
	
	

	return 0;
	
}


//==

