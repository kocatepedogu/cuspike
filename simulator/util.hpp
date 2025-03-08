#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>

#define CUDA_CALL(CALL) \
do { \
    cudaError_t e = CALL; \
    if (e != cudaSuccess) { \
        std::cout << "CUDA runtime error at line" << __LINE__ << ": " << \
        cudaGetErrorString(e) << std::endl; \
        abort(); \
    } \
} while (0);

#endif
