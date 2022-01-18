#include "parallel_solution_v4.cuh"
#include "timer.cuh"

namespace KernelFunction {
    __device__ u_int32_t blockCount;
    __global__ void
    updateSeamMapKernelPipelining(int32_t *input, u_int32_t inputWidth,
                                  bool volatile *isBlockFinished) {
        // 1.  Get block Index
        __shared__ u_int32_t newBlockIdx;
        if (threadIdx.x == 0) {
            newBlockIdx = atomicAdd(&blockCount, 1);
        }
        __syncthreads();

        u_int32_t numBlocksPerRow = ((inputWidth - 1) / blockDim.x + 1);
        u_int32_t currentRow = (newBlockIdx / numBlocksPerRow) + 1;
        u_int32_t currentRowBlock = newBlockIdx % numBlocksPerRow;
        u_int32_t c = currentRowBlock * blockDim.x + threadIdx.x;
        int32_t minVal = 0;

        // 2. Waiting for before block newBlockIdx - numBlocksPerRow
        if (threadIdx.x == 0) {
            if (newBlockIdx >= numBlocksPerRow) {
                while (!isBlockFinished[newBlockIdx - numBlocksPerRow]);
            }
            __threadfence();
        }
        __syncthreads();

        if (c < inputWidth && threadIdx.x != blockDim.x - 1 && threadIdx.x != 0) {
            minVal = input[convertIndex(currentRow - 1, c, inputWidth)];
            if (c > 0)
                minVal = min(minVal, input[convertIndex(currentRow - 1, c - 1, inputWidth)]);
            if (c + 1 < inputWidth)
                minVal = min(minVal, input[convertIndex(currentRow - 1, c + 1, inputWidth)]);
        }

        // 3. Waiting for before block newBlockIdx + 1 and newBlockIdx - 1
        if (threadIdx.x == 0 || threadIdx.x == blockDim.x - 1) {
            if (newBlockIdx > numBlocksPerRow) {
                if (newBlockIdx % numBlocksPerRow != 0) {
                    while (!isBlockFinished[newBlockIdx - numBlocksPerRow - 1]);
                }
                if (newBlockIdx % numBlocksPerRow != numBlocksPerRow - 1) {
                    while (!isBlockFinished[newBlockIdx - numBlocksPerRow + 1]);
                }
                __threadfence();
            }
            if (c < inputWidth) {
                minVal = input[convertIndex(currentRow - 1, c, inputWidth)];
                if (c > 0)
                    minVal = min(minVal, input[convertIndex(currentRow - 1, c - 1, inputWidth)]);
                if (c + 1 < inputWidth)
                    minVal = min(minVal, input[convertIndex(currentRow - 1, c + 1, inputWidth)]);
            }
        }

        if (c < inputWidth)
            input[convertIndex(currentRow, c, inputWidth)] += minVal;

        __syncthreads();

        // 4. Mark Threads as Done
        __threadfence();
        if (threadIdx.x == 0) {
            isBlockFinished[newBlockIdx] = true;
        }
    }
}


PnmImage ParallelSolutionV4::run(const PnmImage &inputImage, int argc, char **argv) {

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
    printf("Running Parallel Solution Version 4 with blockSize=(%d;%d).\n", blockSize.x, blockSize.y);
    GpuTimer timer;
    GpuTimer stepTimer;

    float cal_energy_time = 0;
    float cal_seam_time = 0;
    float extract_seam_time = 0;
    float delete_seam_time = 0;

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
        stepTimer.Start();
        calculateEnergyMap(d_grayImage, inputImage.getWidth() - i, inputImage.getHeight(), d_filterX, d_filterY,
                           FILTER_SIZE, blockSize, d_energyMap);
        stepTimer.Stop();
        cal_energy_time += stepTimer.Elapsed();

        // 2. Dynamic Programming
        stepTimer.Start();
        calculateSeamMap(d_energyMap, inputImage.getWidth() - i, inputImage.getHeight(), blockSize.x * blockSize.y);
        stepTimer.Stop();
        cal_seam_time += stepTimer.Elapsed();

        // 3. Extract the seam
        stepTimer.Start();
        CHECK(cudaMemcpy(energyMap, d_energyMap,
                         (inputImage.getWidth() - i) * inputImage.getHeight() * sizeof(int32_t),
                         cudaMemcpyDeviceToHost))
        extractSeam(energyMap, inputImage.getWidth() - i, inputImage.getHeight(), seam);
        stepTimer.Stop();
        extract_seam_time += stepTimer.Elapsed();

        // 4. Delete the seam
        stepTimer.Start();
        deleteSeam(d_grayImage, inputImage.getWidth() - i, inputImage.getHeight(), seam, blockSize, d_grayImageTemp);
        deleteSeam(d_inputImage, inputImage.getWidth() - i, inputImage.getHeight(), seam, blockSize, d_inputImageTemp);
        stepTimer.Stop();
        delete_seam_time += stepTimer.Elapsed();

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
    printf("Step time: %.3f ms \t %.3f ms \t %.3f ms \t %.3f ms\n", cal_energy_time, cal_seam_time, extract_seam_time, delete_seam_time);
    printf("-------------------------------\n");

    // Return
    return outputImage;
}

void ParallelSolutionV4::calculateSeamMap(int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight,
                                          uint32_t blockSize) {
    uint32_t gridSize = ((inputWidth - 1) / blockSize + 1) * (inputHeight - 1);
    uint32_t zero = 0;

    // Create Device Memory
    bool *isBlockFinished;
    CHECK(cudaMalloc(&isBlockFinished, gridSize * sizeof(bool)))

    // Copy Memory from Host to Device
    CHECK(cudaMemcpyToSymbol(KernelFunction::blockCount, &zero, sizeof(u_int32_t), 0, cudaMemcpyHostToDevice))
    CHECK(cudaMemset(isBlockFinished, 0, gridSize * sizeof(bool)))

    // Run Device Methods
    KernelFunction::updateSeamMapKernelPipelining<<<gridSize, blockSize>>>(d_inputImage, inputWidth, isBlockFinished);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    // Copy Memory from Device to Host

    // Free Device Memory
    CHECK(cudaFree(isBlockFinished))

    // Free Host Memory

    // Return result
}
