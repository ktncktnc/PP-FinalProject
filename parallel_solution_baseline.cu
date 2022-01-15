#include "parallel_solution_baseline.cuh"
#include "timer.cuh"

namespace KernelFunction {
    __device__ __host__  u_int32_t convertIndex(u_int32_t i, u_int32_t j, u_int32_t n) {
        return i * n + j;
    }

    __global__ void
    convertToGrayScaleKernel(const uchar3 *input, u_int32_t inputWidth, u_int32_t inputHeight, int32_t *output) {
        u_int32_t r = blockIdx.y * blockDim.y + threadIdx.y;
        u_int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
        if (r < inputHeight && c < inputWidth) {
            u_int32_t i = r * inputWidth + c;
            uchar3 pixel = input[i];
            output[i] = int32_t(299 * pixel.x + 587 * pixel.y + 114 * pixel.z) / 1000;
        }
    }

    __global__ void
    convolutionKernel(const int32_t *input, u_int32_t inputWidth, u_int32_t inputHeight, const int32_t *filter,
                      u_int32_t filterSize, int32_t *output) {
        u_int32_t out_r = blockIdx.y * blockDim.y + threadIdx.y;
        u_int32_t out_c = blockIdx.x * blockDim.x + threadIdx.x;
        if (out_c >= inputWidth || out_r >= inputHeight) return;
        int32_t outPixel = 0;
        for (int32_t k_r = -int(filterSize / 2); k_r <= int(filterSize / 2); ++k_r) {
            u_int32_t in_r = min(max(0, k_r + int(out_r)), inputHeight - 1);
            for (int32_t k_c = -int(filterSize / 2); k_c <= int(filterSize / 2); ++k_c) {
                uint32_t in_c = min(max(0, k_c + int(out_c)), inputWidth - 1);
                int32_t inPixel = input[convertIndex(in_r, in_c, inputWidth)];
                int32_t filterVal = filter[convertIndex(k_r + filterSize / 2, k_c + filterSize / 2, filterSize)];
                outPixel += inPixel * filterVal;
            }
        }
        output[convertIndex(out_r, out_c, inputWidth)] = outPixel;
    }

    __global__ void
    addAbsKernel(const int32_t *input_1, const int32_t *input_2, u_int32_t inputWidth, u_int32_t inputHeight,
                 int32_t *output) {
        u_int32_t r = blockIdx.y * blockDim.y + threadIdx.y;
        u_int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
        if (r < inputHeight && c < inputWidth) {
            u_int32_t i = r * inputWidth + c;
            output[i] = abs(input_1[i]) + abs(input_2[i]);
        }
    }

    __global__ void
    updateSeamMapKernel(int32_t *input, u_int32_t inputWidth,
                        int32_t currentRow) {
        u_int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < inputWidth) {
            int32_t minVal = input[convertIndex(currentRow - 1, c, inputWidth)];
            if (c > 0)
                minVal = min(minVal, input[convertIndex(currentRow - 1, c - 1, inputWidth)]);
            if (c + 1 < inputWidth)
                minVal = min(minVal, input[convertIndex(currentRow - 1, c + 1, inputWidth)]);
            input[convertIndex(currentRow, c, inputWidth)] += minVal;
        }
    }

    __global__ void
    deleteSeamKernel(const uchar3 *input, u_int32_t inputWidth, u_int32_t inputHeight, const u_int32_t *seam,
                     uchar3 *output) {
        u_int32_t r = blockIdx.y * blockDim.y + threadIdx.y;
        u_int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
        if (r < inputHeight && c + 1 < inputWidth) {
            u_int32_t inputC = c;
            if (inputC >= seam[r])
                inputC++;
            output[convertIndex(r, c, inputWidth - 1)] = input[convertIndex(r, inputC, inputWidth)];
        }
    }
}

const int32_t ParallelSolutionBaseline::SOBEL_X[3][3] = {{1, 0, -1},
                                                         {2, 0, -2},
                                                         {1, 0, -1}};
const int32_t ParallelSolutionBaseline::SOBEL_Y[3][3] = {{1,  2,  1},
                                                         {0,  0,  0},
                                                         {-1, -2, -1}};

PnmImage ParallelSolutionBaseline::run(const PnmImage &inputImage, int argc, char **argv) {
    // Extract arguments
    int nDeletingSeams = 1;
    dim3 blockSize(32, 32); // Default
    if (argc > 0)
        nDeletingSeams = int(strtol(argv[0], nullptr, 10));
    if (argc > 1) {
        blockSize.x = strtol(argv[1], nullptr, 10);
        blockSize.y = strtol(argv[2], nullptr, 10);
    }
    printf("Running Baseline Parallel Solution with blockSize=(%d;%d).\n", blockSize.x, blockSize.y);
    GpuTimer timer;
    timer.Start();

    PnmImage outputImage = inputImage;
    for (int i = 0; i < nDeletingSeams; ++i) {
        // 1. Convert to GrayScale
        IntImage grayImage = convertToGrayScale(outputImage, blockSize);
        // 2. Calculate the Energy Map
        IntImage energyMap = calculateEnergyMap(grayImage, blockSize);
        // 3. Dynamic Programming
        IntImage seamMap = calculateSeamMap(energyMap, blockSize.x * blockSize.y);
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

IntImage ParallelSolutionBaseline::calculateEnergyMap(const IntImage &inputImage, dim3 blockSize) {
    // Create Host Memory
    dim3 gridSize((inputImage.getWidth() - 1) / blockSize.x + 1, (inputImage.getHeight() - 1) / blockSize.y + 1);
    IntImage outputImage = IntImage(inputImage.getWidth(), inputImage.getHeight());

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
    KernelFunction::convolutionKernel<<<gridSize, blockSize>>>(d_inputImage, inputImage.getWidth(), inputImage.getHeight(), d_filterX, FILTER_SIZE, d_outputImageX);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    KernelFunction::convolutionKernel<<<gridSize, blockSize>>>(d_inputImage, inputImage.getWidth(), inputImage.getHeight(), d_filterY, FILTER_SIZE, d_outputImageY);
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

IntImage ParallelSolutionBaseline::convertToGrayScale(const PnmImage &inputImage, dim3 blockSize) {
    // Create Host Memory
    dim3 gridSize((inputImage.getWidth() - 1) / blockSize.x + 1, (inputImage.getHeight() - 1) / blockSize.y + 1);
    IntImage outputImage = IntImage(inputImage.getWidth(), inputImage.getHeight());

    // Create Device Memory
    uchar3 *d_inputImage;
    CHECK(cudaMalloc(&d_inputImage, inputImage.getWidth() * inputImage.getHeight() * sizeof(uchar3)))
    int32_t *d_outputImage;
    CHECK(cudaMalloc(&d_outputImage, outputImage.getWidth() * outputImage.getHeight() * sizeof(int32_t)))

    // Copy Memory from Host to Device
    CHECK(cudaMemcpy(d_inputImage, inputImage.getPixels(),
                     inputImage.getWidth() * inputImage.getHeight() * sizeof(uchar3), cudaMemcpyHostToDevice))

    // Run Device Methods
    KernelFunction::convertToGrayScaleKernel<<<gridSize, blockSize>>>(d_inputImage, inputImage.getWidth(), inputImage.getHeight(), d_outputImage);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    // Copy Memory from Device to Host
    CHECK(cudaMemcpy(outputImage.getPixels(), d_outputImage,
                     outputImage.getWidth() * outputImage.getHeight() * sizeof(int32_t), cudaMemcpyDeviceToHost))

    // Free Device Memory
    CHECK(cudaFree(d_inputImage))
    CHECK(cudaFree(d_outputImage))

    // Free Host Memory

    // Return result
    return outputImage;
}

IntImage ParallelSolutionBaseline::calculateSeamMap(const IntImage &inputImage, uint32_t blockSize) {
    // Create Host Memory
    uint32_t gridSize = (inputImage.getWidth() - 1) / blockSize + 1;
    IntImage outputImage = IntImage(inputImage.getWidth(), inputImage.getHeight());

    // Create Device Memory
    int32_t *d_inputImage;
    CHECK(cudaMalloc(&d_inputImage, inputImage.getWidth() * inputImage.getHeight() * sizeof(int32_t)))

    // Copy Memory from Host to Device
    CHECK(cudaMemcpy(d_inputImage, inputImage.getPixels(),
                     inputImage.getWidth() * inputImage.getHeight() * sizeof(int32_t), cudaMemcpyHostToDevice))

    // Run Device Methods
    for (int i = 1; i < inputImage.getHeight(); ++i) {
        KernelFunction::updateSeamMapKernel<<<gridSize, blockSize>>>(d_inputImage, inputImage.getWidth(), i);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError())
    }

    // Copy Memory from Device to Host
    CHECK(cudaMemcpy(outputImage.getPixels(), d_inputImage,
                     outputImage.getWidth() * outputImage.getHeight() * sizeof(int32_t), cudaMemcpyDeviceToHost))

    // Free Device Memory
    CHECK(cudaFree(d_inputImage))

    // Free Host Memory

    // Return result
    return outputImage;
}

void ParallelSolutionBaseline::extractSeam(const IntImage &energyMap, uint32_t *seam) {
    // Find minSeam
    u_int32_t minValC = 0;
    for (int c = 1; c < energyMap.getWidth(); ++c)
        if (energyMap.getPixels()[KernelFunction::convertIndex(energyMap.getHeight() - 1, c, energyMap.getWidth())] <
            energyMap.getPixels()[KernelFunction::convertIndex(energyMap.getHeight() - 1, minValC,
                                                               energyMap.getWidth())]) {
            minValC = c;
        }
    // Trace back
    seam[energyMap.getHeight() - 1] = minValC;
    for (int r = int(energyMap.getHeight() - 2); r >= 0; r--) {
        auto c = minValC;
        if (c > 0) {
            if (energyMap.getPixels()[KernelFunction::convertIndex(r, c - 1, energyMap.getWidth())] <=
                energyMap.getPixels()[KernelFunction::convertIndex(r, minValC, energyMap.getWidth())]) {
                minValC = c - 1;
            }
        }
        if (c + 1 < energyMap.getWidth()) {
            if (energyMap.getPixels()[KernelFunction::convertIndex(r, c + 1, energyMap.getWidth())] <
                energyMap.getPixels()[KernelFunction::convertIndex(r, minValC, energyMap.getWidth())]) {
                minValC = c + 1;
            }
        }
        seam[r] = minValC;
    }
}

PnmImage ParallelSolutionBaseline::deleteSeam(const PnmImage &inputImage, uint32_t *seam, dim3 blockSize) {
    // Create Host Memory
    dim3 gridSize((inputImage.getWidth() - 1) / blockSize.x + 1, (inputImage.getHeight() - 1) / blockSize.y + 1);
    PnmImage outputImage = PnmImage(inputImage.getWidth() - 1, inputImage.getHeight());

    // Create Device Memory
    uchar3 *d_inputImage;
    CHECK(cudaMalloc(&d_inputImage, inputImage.getWidth() * inputImage.getHeight() * sizeof(uchar3)))
    u_int32_t *d_seam;
    CHECK(cudaMalloc(&d_seam, outputImage.getHeight() * sizeof(u_int32_t)))
    uchar3 *d_outputImage;
    CHECK(cudaMalloc(&d_outputImage, outputImage.getWidth() * outputImage.getHeight() * sizeof(uchar3)))

    // Copy Memory from Host to Device
    CHECK(cudaMemcpy(d_inputImage, inputImage.getPixels(),
                     inputImage.getWidth() * inputImage.getHeight() * sizeof(uchar3), cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(d_seam, seam,
                     inputImage.getHeight() * sizeof(u_int32_t), cudaMemcpyHostToDevice))

    // Run Device Methods
    KernelFunction::deleteSeamKernel<<<gridSize, blockSize>>>(d_inputImage, inputImage.getWidth(), inputImage.getHeight(), d_seam, d_outputImage);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    // Copy Memory from Device to Host
    CHECK(cudaMemcpy(outputImage.getPixels(), d_outputImage,
                     outputImage.getWidth() * outputImage.getHeight() * sizeof(uchar3), cudaMemcpyDeviceToHost))

    // Free Device Memory
    CHECK(cudaFree(d_inputImage))
    CHECK(cudaFree(d_seam))
    CHECK(cudaFree(d_outputImage))

    // Free Host Memory

    // Return result
    return outputImage;
}


