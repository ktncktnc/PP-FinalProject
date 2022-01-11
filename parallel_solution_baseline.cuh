#pragma once

#include "solution.cuh"
#include "utils.cuh"

namespace KernelFunction {
    __device__ __host__ u_int32_t convertIndex(u_int32_t i, u_int32_t j, u_int32_t n);

    __global__ void
    convertToGrayScaleKernel(const uchar3 *input, u_int32_t inputWidth, u_int32_t inputHeight, int32_t *output);

    __global__ void
    convolutionKernel(const int32_t *input, u_int32_t inputWidth, u_int32_t inputHeight, const int32_t *filter,
                      u_int32_t filterSize, int32_t *output);

    __global__ void
    addAbsKernel(const int32_t *input_1, const int32_t *input_2, u_int32_t inputWidth, u_int32_t inputHeight,
                 int32_t *output);

    __global__ void
    updateSeamMapKernel(int32_t *input, u_int32_t inputWidth,
                        int32_t currentRow);

    __global__ void
    deleteSeamKernel(const uchar3 *input, u_int32_t inputWidth, u_int32_t inputHeight, const u_int32_t * seam, uchar3 *output);

}

class ParallelSolutionBaseline : public BaseSolution {
private:
    static const u_int32_t FILTER_SIZE = 3;
    static const int32_t SOBEL_X[3][3];
    static const int32_t SOBEL_Y[3][3];

    static IntImage convertToGrayScale(const PnmImage &inputImage, dim3 blockSize = dim3(32, 32));

    static IntImage calculateEnergyMap(const IntImage &inputImage, dim3 blockSize = dim3(32, 32));

    static IntImage calculateSeamMap(const IntImage &inputImage, uint32_t blockSize = 32);

    static void extractSeam(const IntImage &energyMap, uint32_t *seam);

    static PnmImage deleteSeam(const PnmImage &inputImage, uint32_t *seam, dim3 blockSize = dim3(32, 32));

public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;
};
