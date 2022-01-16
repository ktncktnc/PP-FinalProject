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
    deleteSeamKernel(const uchar3 *input, u_int32_t inputWidth, u_int32_t inputHeight, const u_int32_t *seam,
                     uchar3 *output);

    __global__ void
    deleteSeamKernel(const int32_t *input, u_int32_t inputWidth, u_int32_t inputHeight, const u_int32_t *seam,
                     int32_t *output);

}

class ParallelSolutionBaseline : public BaseSolution {
public:
    static const u_int32_t FILTER_SIZE = 3;
protected:
    static const int32_t SOBEL_X[3][3];
    static const int32_t SOBEL_Y[3][3];

    static void convertToGrayScale(const uchar3 *d_inputImage, uint32_t inputWidth, uint32_t inputHeight,
                                   dim3 blockSize, int32_t *d_outputImage);

    static void
    calculateEnergyMap(const int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight, const int32_t *d_filterX,
                       const int32_t *d_filterY, uint32_t filterSize, dim3 blockSize, int32_t *d_outputImage);

    static void calculateSeamMap(int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight, uint32_t blockSize);

    static void extractSeam(const int32_t *energyMap, uint32_t inputWidth, uint32_t inputHeight, uint32_t *seam);

    static void deleteSeam(const int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight, const uint32_t *seam,
                           dim3 blockSize, int32_t *d_outputImage);

    static void deleteSeam(const uchar3 *d_inputImage, uint32_t inputWidth, uint32_t inputHeight, const uint32_t *seam,
                           dim3 blockSize, uchar3 *d_outputImage);

    static void swap(int32_t *&pa, int32_t *&pb);

    static void swap(uchar3 *&pa, uchar3 *&pb);

public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;
};
