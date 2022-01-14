//
// Created by phuc on 12/01/2022.
//

#include "timer.cuh"
#include "parallel_solution_v2.cuh"

namespace KernelFunction {
    __global__ void
    convolutionKernel_v2(const int32_t *input, u_int32_t width, u_int32_t height, const int32_t *filter,
                         u_int32_t filterSize, int32_t *output) {
        extern __shared__ int32_t s_input[];

        u_int32_t s_width = blockDim.x + filterSize - 1;
        u_int32_t s_height = blockDim.y + filterSize - 1;

        u_int32_t out_r = blockIdx.y * blockDim.y + threadIdx.y;
        u_int32_t out_c = blockIdx.x * blockDim.x + threadIdx.x;
        u_int32_t out_idx = convertIndex(out_r, out_c, width);

        u_int32_t sharedGridSize = (s_width * s_height) / (blockDim.x * blockDim.y);
        for (int i = 0; i <= sharedGridSize; i++) {
            u_int32_t shared_idx = threadIdx.y * blockDim.x + threadIdx.x + i * (blockDim.x * blockDim.y);
            u_int32_t shared_r = shared_idx / s_width;
            u_int32_t shared_c = shared_idx % s_width;

            int32_t gIdx_r = (int32_t) shared_r - int32_t(filterSize) / 2 + int32_t(blockIdx.y * blockDim.y);
            int32_t gIdx_c = (int32_t) shared_c - int32_t(filterSize) / 2 + int32_t(blockIdx.x * blockDim.x);
            gIdx_c = max(0, min(int32_t(width) - 1, gIdx_c));
            gIdx_r = max(0, min(int32_t(height) - 1, gIdx_r));
            uint32_t g_idx = convertIndex(gIdx_r, gIdx_c, width);

            if (shared_c < s_width && shared_r < s_height)
                s_input[shared_idx] = input[g_idx];
        }

        __syncthreads();

        if (out_c >= width || out_r >= height) return;
        int32_t outPixel = 0;

        for (int32_t k_r = -int(filterSize / 2); k_r <= int(filterSize / 2); ++k_r) {
            u_int32_t in_r = threadIdx.y + filterSize / 2 + k_r;

            for (int32_t k_c = -int(filterSize / 2); k_c <= int(filterSize / 2); ++k_c) {
                uint32_t in_c = threadIdx.x + filterSize / 2 + k_c;

                int32_t inPixel = s_input[convertIndex(in_r, in_c, s_width)];
                int32_t filterVal = filter[convertIndex(k_r + filterSize / 2, k_c + filterSize / 2, filterSize)];
                outPixel += inPixel * filterVal;
            }
        }
        output[out_idx] = outPixel;
    }
}

IntImage ParallelSolutionV2::calculateEnergyMap(const IntImage &inputImage, dim3 blockSize) {
    // Create Host Memory
    dim3 gridSize((inputImage.getWidth() - 1) / blockSize.x + 1, (inputImage.getHeight() - 1) / blockSize.y + 1);
    IntImage outputImage = IntImage(inputImage.getWidth(), inputImage.getHeight());
    size_t smemSize = (blockSize.x + FILTER_SIZE - 1) * (blockSize.y + FILTER_SIZE - 1) * sizeof(int32_t);

    // Create Device Memory
    int32_t *d_filterX;
    CHECK(cudaMalloc(&d_filterX, FILTER_SIZE * FILTER_SIZE * sizeof(int32_t)))
    int32_t *d_filterY;
    CHECK(cudaMalloc(&d_filterY, FILTER_SIZE * FILTER_SIZE * sizeof(int32_t)))
    int32_t *d_inputImage;
    CHECK(cudaMalloc(&d_inputImage, inputImage.getWidth() * inputImage.getHeight() * sizeof(int32_t)))
    int32_t *d_outputImageX;
    CHECK(cudaMalloc(&d_outputImageX, outputImage.getWidth() * outputImage.getHeight() * sizeof(int32_t)))
    int32_t *d_outputImageY;
    CHECK(cudaMalloc(&d_outputImageY, outputImage.getWidth() * outputImage.getHeight() * sizeof(int32_t)))
    int32_t *d_outputImage;
    CHECK(cudaMalloc(&d_outputImage, outputImage.getWidth() * outputImage.getHeight() * sizeof(int32_t)))

    // Copy Memory from Host to Device
    CHECK(cudaMemcpy(d_filterX, SOBEL_X, FILTER_SIZE * FILTER_SIZE * sizeof(int32_t), cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(d_filterY, SOBEL_Y, FILTER_SIZE * FILTER_SIZE * sizeof(int32_t), cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(d_inputImage, inputImage.getPixels(),
                     inputImage.getWidth() * inputImage.getHeight() * sizeof(int32_t), cudaMemcpyHostToDevice))

    // Run Device Methods
    KernelFunction::convolutionKernel_v2<<<gridSize, blockSize, smemSize>>>(d_inputImage, inputImage.getWidth(), inputImage.getHeight(), d_filterX, FILTER_SIZE, d_outputImageX);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    KernelFunction::convolutionKernel_v2<<<gridSize, blockSize, smemSize>>>(d_inputImage, inputImage.getWidth(), inputImage.getHeight(), d_filterY, FILTER_SIZE, d_outputImageY);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    KernelFunction::addAbsKernel<<<gridSize, blockSize>>>(d_outputImageX, d_outputImageY, outputImage.getWidth(), outputImage.getHeight(), d_outputImage);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())


    // Copy Memory from Device to Host
    CHECK(cudaMemcpy(outputImage.getPixels(), d_outputImage,
                     outputImage.getWidth() * outputImage.getHeight() * sizeof(int32_t), cudaMemcpyDeviceToHost))

    // Free Device Memory
    CHECK(cudaFree(d_filterX))
    CHECK(cudaFree(d_filterY))
    CHECK(cudaFree(d_inputImage))
    CHECK(cudaFree(d_outputImageX))
    CHECK(cudaFree(d_outputImageY))
    CHECK(cudaFree(d_outputImage))

    // Free Host Memory

    // Return result
    return outputImage;
}

PnmImage ParallelSolutionV2::run(const PnmImage &inputImage, int argc, char **argv) {
    // Extract arguments
    int nDeletingSeams = 1;
    dim3 blockSize(32, 32); // Default
    if (argc > 0)
        nDeletingSeams = int(strtol(argv[0], nullptr, 10));
    if (argc > 1) {
        blockSize.x = strtol(argv[1], nullptr, 10);
        blockSize.y = strtol(argv[2], nullptr, 10);
    }
    printf("Running Parallel Solution Version 2 with blockSize=(%d;%d).\n", blockSize.x, blockSize.y);
    GpuTimer timer;
    timer.Start();

    PnmImage outputImage = inputImage;
    for (int i = 0; i < nDeletingSeams; ++i) {
        // 1. Convert to GrayScale
        IntImage grayImage = convertToGrayScale(outputImage, blockSize);
        // 2. Calculate the Energy Map
        IntImage energyMap = calculateEnergyMap(grayImage, blockSize);
        // 3. Dynamic Programming
        IntImage seamMap = calculateSeamMap(energyMap, blockSize.x);
        // 4. Extract the seam
        auto *seam = (uint32_t *) malloc(energyMap.getHeight() * sizeof(uint32_t));
        extractSeam(seamMap, seam);
        // 5. Delete the seam
        outputImage = deleteSeam(outputImage, seam);
        free(seam);
    }
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
    printf("-------------------------------\n");
    return outputImage;
}