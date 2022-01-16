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

    __global__ void
    deleteSeamKernel(const int32_t *input, u_int32_t inputWidth, u_int32_t inputHeight, const u_int32_t *seam,
                     int32_t *output) {
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

    // Start Timer
    printf("Running Baseline Parallel Solution with blockSize=(%d;%d).\n", blockSize.x, blockSize.y);
    GpuTimer timer;
    timer.Start();

    // Create Host Variable
    PnmImage outputImage(inputImage.getWidth() - nDeletingSeams, inputImage.getHeight());

    // Create Host Memory
    auto *seam = (uint32_t *) malloc(inputImage.getHeight() * sizeof(uint32_t));
    auto *energyMap = (int32_t *) malloc(inputImage.getHeight() * inputImage.getWidth() * sizeof(int32_t));

    // Create Device Memory
    uchar3 *d_inputImage;
    CHECK(cudaMalloc(&d_inputImage, inputImage.getWidth() * inputImage.getHeight() * sizeof(uchar3)))
    uchar3 *d_inputImageTemp;
    CHECK(cudaMalloc(&d_inputImageTemp, inputImage.getWidth() * inputImage.getHeight() * sizeof(uchar3)))
    int32_t *d_grayImage;
    CHECK(cudaMalloc(&d_grayImage, inputImage.getWidth() * inputImage.getHeight() * sizeof(int32_t)))
    int32_t *d_grayImageTemp;
    CHECK(cudaMalloc(&d_grayImageTemp, inputImage.getWidth() * inputImage.getHeight() * sizeof(int32_t)))
    int32_t *d_energyMap;
    CHECK(cudaMalloc(&d_energyMap, inputImage.getWidth() * inputImage.getHeight() * sizeof(int32_t)))
    int32_t *d_filterX;
    CHECK(cudaMalloc(&d_filterX, FILTER_SIZE * FILTER_SIZE * sizeof(int32_t)))
    int32_t *d_filterY;
    CHECK(cudaMalloc(&d_filterY, FILTER_SIZE * FILTER_SIZE * sizeof(int32_t)))

    // Copy Memory from Host to Device
    CHECK(cudaMemcpy(d_inputImage, inputImage.getPixels(),
                     inputImage.getWidth() * inputImage.getHeight() * sizeof(uchar3), cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(d_filterX, SOBEL_X, FILTER_SIZE * FILTER_SIZE * sizeof(int32_t), cudaMemcpyHostToDevice))
    CHECK(cudaMemcpy(d_filterY, SOBEL_Y, FILTER_SIZE * FILTER_SIZE * sizeof(int32_t), cudaMemcpyHostToDevice))

    // Run Kernel functions
    convertToGrayScale(d_inputImage, inputImage.getWidth(), inputImage.getHeight(), blockSize, d_grayImage);
    for (int i = 0; i < nDeletingSeams; ++i) {
        // 1. Calculate the Energy Map
        calculateEnergyMap(d_grayImage, inputImage.getWidth() - i, inputImage.getHeight(), d_filterX, d_filterY,
                           FILTER_SIZE, blockSize, d_energyMap);
        // 2. Dynamic Programming
        calculateSeamMap(d_energyMap, inputImage.getWidth() - i, inputImage.getHeight(), blockSize.x * blockSize.y);
        // 3. Extract the seam
        CHECK(cudaMemcpy(energyMap, d_energyMap,
                         (inputImage.getWidth() - i) * inputImage.getHeight() * sizeof(int32_t),
                         cudaMemcpyDeviceToHost));
        extractSeam(energyMap, inputImage.getWidth() - i, inputImage.getHeight(), seam);
        // 4. Delete the seam
        deleteSeam(d_grayImage, inputImage.getWidth() - i, inputImage.getHeight(), seam, blockSize, d_grayImageTemp);
        deleteSeam(d_inputImage, inputImage.getWidth() - i, inputImage.getHeight(), seam, blockSize, d_inputImageTemp);
        swap(d_grayImage, d_grayImageTemp);
        swap(d_inputImage, d_inputImageTemp);
    }

    // Copy memory from device to host
    CHECK(cudaMemcpy(outputImage.getPixels(), d_inputImage,
                     outputImage.getWidth() * outputImage.getHeight() * sizeof(uchar3), cudaMemcpyDeviceToHost))

    // Free Device Memory
    CHECK(cudaFree(d_inputImage))
    CHECK(cudaFree(d_inputImageTemp))
    CHECK(cudaFree(d_grayImage))
    CHECK(cudaFree(d_grayImageTemp))
    CHECK(cudaFree(d_energyMap))
    CHECK(cudaFree(d_filterX))
    CHECK(cudaFree(d_filterY))

    // Free Host Memory
    free(seam);
    free(energyMap);

    // Stop Timer
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
    printf("-------------------------------\n");

    // Return
    return outputImage;
}

void ParallelSolutionBaseline::convertToGrayScale(const uchar3 *d_inputImage, uint32_t inputWidth, uint32_t inputHeight,
                                                  dim3 blockSize, int32_t *d_outputImage) {
    // Create Host Memory
    dim3 gridSize((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);

    // Copy Memory from Host to Device

    // Run Device Methods
    KernelFunction::convertToGrayScaleKernel<<<gridSize, blockSize>>>(d_inputImage, inputWidth, inputHeight, d_outputImage);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    // Copy Memory from Device to Host

    // Free Device Memory

    // Free Host Memory

    // Return result
}

void
ParallelSolutionBaseline::calculateEnergyMap(const int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight,
                                             const int32_t *d_filterX, const int32_t *d_filterY, uint32_t filterSize,
                                             dim3 blockSize, int32_t *d_outputImage) {
    // Create Host Memory
    dim3 gridSize((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);

    // Create Device Memory
    int32_t *d_outputImageX;
    CHECK(cudaMalloc(&d_outputImageX, inputWidth * inputHeight * sizeof(int32_t)))
    int32_t *d_outputImageY;
    CHECK(cudaMalloc(&d_outputImageY, inputWidth * inputHeight * sizeof(int32_t)))

    // Copy Memory from Host to Device

    // Run Device Methods
    KernelFunction::convolutionKernel<<<gridSize, blockSize>>>(d_inputImage, inputWidth, inputHeight, d_filterX, filterSize, d_outputImageX);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    KernelFunction::convolutionKernel<<<gridSize, blockSize>>>(d_inputImage, inputWidth, inputHeight, d_filterY, filterSize, d_outputImageY);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    KernelFunction::addAbsKernel<<<gridSize, blockSize>>>(d_outputImageX, d_outputImageY, inputWidth, inputHeight, d_outputImage);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    // Copy Memory from Device to Host

    // Free Device Memory
    CHECK(cudaFree(d_outputImageX))
    CHECK(cudaFree(d_outputImageY))

    // Free Host Memory

    // Return result
}

void ParallelSolutionBaseline::calculateSeamMap(int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight,
                                                uint32_t blockSize) {
    // Create Host Memory
    uint32_t gridSize = (inputWidth - 1) / blockSize + 1;

    // Create Device Memory

    // Copy Memory from Host to Device

    // Run Device Methods
    for (int i = 1; i < inputHeight; ++i) {
        KernelFunction::updateSeamMapKernel<<<gridSize, blockSize>>>(d_inputImage, inputWidth, i);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError())
    }

    // Copy Memory from Device to Host

    // Free Device Memory

    // Free Host Memory

    // Return result
}

void
ParallelSolutionBaseline::extractSeam(const int32_t *energyMap, uint32_t inputWidth, uint32_t inputHeight,
                                      uint32_t *seam) {
    // Find minSeam
    u_int32_t minValC = 0;
    for (int c = 1; c < inputWidth; ++c)
        if (energyMap[KernelFunction::convertIndex(inputHeight - 1, c, inputWidth)] <
            energyMap[KernelFunction::convertIndex(inputHeight - 1, minValC,
                                                   inputWidth)]) {
            minValC = c;
        }
    // Trace back
    seam[inputHeight - 1] = minValC;
    for (int r = int(inputHeight - 2); r >= 0; r--) {
        auto c = minValC;
        if (c > 0) {
            if (energyMap[KernelFunction::convertIndex(r, c - 1, inputWidth)] <=
                energyMap[KernelFunction::convertIndex(r, minValC, inputWidth)]) {
                minValC = c - 1;
            }
        }
        if (c + 1 < inputWidth) {
            if (energyMap[KernelFunction::convertIndex(r, c + 1, inputWidth)] <
                energyMap[KernelFunction::convertIndex(r, minValC, inputWidth)]) {
                minValC = c + 1;
            }
        }
        seam[r] = minValC;
    }
}

void ParallelSolutionBaseline::deleteSeam(const int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight,
                                          const uint32_t *seam, dim3 blockSize, int32_t *d_outputImage) {
    // Create Host Memory
    dim3 gridSize((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);

    // Create Device Memory
    u_int32_t *d_seam;
    CHECK(cudaMalloc(&d_seam, inputHeight * sizeof(u_int32_t)))

    // Copy Memory from Host to Device
    CHECK(cudaMemcpy(d_seam, seam,
                     inputHeight * sizeof(u_int32_t), cudaMemcpyHostToDevice))

    // Run Device Methods
    KernelFunction::deleteSeamKernel<<<gridSize, blockSize>>>(d_inputImage, inputWidth, inputHeight, d_seam, d_outputImage);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    // Copy Memory from Device to Host

    // Free Device Memory
    CHECK(cudaFree(d_seam))

    // Free Host Memory

    // Return result
}

void ParallelSolutionBaseline::deleteSeam(const uchar3 *d_inputImage, uint32_t inputWidth, uint32_t inputHeight,
                                          const uint32_t *seam, dim3 blockSize, uchar3 *d_outputImage) {
    // Create Host Memory
    dim3 gridSize((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);

    // Create Device Memory
    u_int32_t *d_seam;
    CHECK(cudaMalloc(&d_seam, inputHeight * sizeof(u_int32_t)))

    // Copy Memory from Host to Device
    CHECK(cudaMemcpy(d_seam, seam,
                     inputHeight * sizeof(u_int32_t), cudaMemcpyHostToDevice))

    // Run Device Methods
    KernelFunction::deleteSeamKernel<<<gridSize, blockSize>>>(d_inputImage, inputWidth, inputHeight, d_seam, d_outputImage);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    // Copy Memory from Device to Host

    // Free Device Memory
    CHECK(cudaFree(d_seam))

    // Free Host Memory

    // Return result
}

void ParallelSolutionBaseline::swap(int32_t *&pa, int32_t *&pb) {
    auto temp = pa;
    pa = pb;
    pb = temp;
}

void ParallelSolutionBaseline::swap(uchar3 *&pa, uchar3 *&pb) {
    auto temp = pa;
    pa = pb;
    pb = temp;
}






