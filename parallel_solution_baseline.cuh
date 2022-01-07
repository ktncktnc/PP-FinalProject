#pragma once

#include "solution.cuh"
#include "utils.cuh"

namespace KernelFunction {
    __global__ void
    scanKernel(const uchar3 *input, u_int32_t inputWidth, u_int32_t inputHeight, const int3 *filter,
               u_int32_t filterSize, int3 *output);

    __global__ void
    addAbsKernel(const int3 *input_1, const int3 *input_2, u_int32_t inputWidth, u_int32_t inputHeight,
                 int3 *output);
}

class ParallelSolutionBaseline : public BaseSolution {
private:
    static const u_int32_t FILTER_SIZE = 3;
    static const int32_t SOBEL_X[3][3];
    static const int32_t SOBEL_Y[3][3];

    IntImage scan(const PnmImage &inputImage, dim3 blockSize = dim3(32, 32));

public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;
};
