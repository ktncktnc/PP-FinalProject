#include "parallel_solution_baseline.cuh"

namespace KernelFunction {
    __global__ void
    scanKernel(const uchar3 *input, u_int32_t inputWidth, u_int32_t inputHeight, const int3 *filter,
               u_int32_t filterSize, int3 *output) {
    }

    __global__ void
    addAbsKernel(const int3 *input_1, const int3 *input_2, u_int32_t inputWidth, u_int32_t inputHeight,
                 int3 *output) {}
}

const int32_t ParallelSolutionBaseline::SOBEL_X[3][3] = {{1, 0, -1},
                                                         {2, 0, -2},
                                                         {1, 0, -1}};
const int32_t ParallelSolutionBaseline::SOBEL_Y[3][3] = {{1,  2,  1},
                                                         {0,  0,  0},
                                                         {-1, -2, -1}};

PnmImage ParallelSolutionBaseline::run(const PnmImage &inputImage, int argc, char **argv) {
    // 1. Scan
    IntImage scannedImage = this->scan(inputImage);
    return BaseSolution::run(inputImage, argc, argv);
}

IntImage ParallelSolutionBaseline::scan(const PnmImage &inputImage, dim3 blockSize) {
    // Create Host Memory
    dim3 gridSize((inputImage.getWidth() - 1) / blockSize.x + 1, (inputImage.getHeight() - 1) / blockSize.y + 1);
    IntImage outputImage = IntImage(inputImage.getWidth(), inputImage.getHeight());

    // Create Device Memory
    int3 *d_filterX;
    CHECK(cudaMalloc(&d_filterX, FILTER_SIZE * FILTER_SIZE * sizeof(int3)));
    int3 *d_filterY;
    CHECK(cudaMalloc(&d_filterY, FILTER_SIZE * FILTER_SIZE * sizeof(int3)))
    uchar3 *d_inputImage;
    CHECK(cudaMalloc(&d_inputImage, inputImage.getWidth() * inputImage.getHeight() * sizeof(uchar3)));
    int3 *d_outputImageX;
    CHECK(cudaMalloc(&d_outputImageX, outputImage.getWidth() * outputImage.getHeight() * sizeof(int3)));
    int3 *d_outputImageY;
    CHECK(cudaMalloc(&d_outputImageY, outputImage.getWidth() * outputImage.getHeight() * sizeof(int3)));
    int3 *d_outputImage;
    CHECK(cudaMalloc(&d_outputImage, outputImage.getWidth() * outputImage.getHeight() * sizeof(int3)));

    // Copy Memory from Host to Device
    CHECK(cudaMemcpy(d_filterX, SOBEL_X, FILTER_SIZE * FILTER_SIZE * sizeof(int3), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filterY, SOBEL_Y, FILTER_SIZE * FILTER_SIZE * sizeof(int3), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_inputImage, inputImage.getPixels(),
                     inputImage.getWidth() * inputImage.getHeight() * sizeof(uchar3), cudaMemcpyHostToDevice));

    // Run Device Methods
    KernelFunction::scanKernel<<<gridSize, blockSize>>>(d_inputImage, inputImage.getWidth(), inputImage.getHeight(), d_filterX,
            FILTER_SIZE, d_outputImageX);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    KernelFunction::scanKernel<<<gridSize, blockSize>>>(d_inputImage, inputImage.getWidth(), inputImage.getHeight(), d_filterY,
            FILTER_SIZE, d_outputImageY);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    KernelFunction::addAbsKernel<<<gridSize, blockSize>>>(d_outputImageX, d_outputImageY, outputImage.getWidth(),
            outputImage.getHeight(), d_outputImage);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy Memory from Device to Host
    CHECK(cudaMemcpy(outputImage.getPixels(), d_outputImage,
                     outputImage.getWidth() * outputImage.getHeight() * sizeof(int3), cudaMemcpyDeviceToHost));

    // Free Device Memory
    CHECK(cudaFree(d_filterX));
    CHECK(cudaFree(d_filterY));
    CHECK(cudaFree(d_inputImage));
    CHECK(cudaFree(d_outputImageX));
    CHECK(cudaFree(d_outputImageY));
    CHECK(cudaFree(d_outputImage));

    // Free Host Memory

    // Return result
    return outputImage;
}


