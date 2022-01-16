#include "parallel_solution_v4.cuh"
#include "timer.cuh"

namespace KernelFunction {
    __device__ u_int32_t blockCount = 0;

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
                while (!isBlockFinished[newBlockIdx - numBlocksPerRow]) {};
            }
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
                    while (!isBlockFinished[newBlockIdx - numBlocksPerRow - 1]) {};
                }
                if (newBlockIdx % numBlocksPerRow != numBlocksPerRow - 1) {
                    while (!isBlockFinished[newBlockIdx - numBlocksPerRow + 1]) {};
                }
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
    printf("Running Parallel Solution Version 4 with blockSize=(%d;%d).\n", blockSize.x, blockSize.y);
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

IntImage ParallelSolutionV4::calculateSeamMap(const IntImage &inputImage, uint32_t blockSize) {
    uint32_t gridSize = ((inputImage.getWidth() - 1) / blockSize + 1) * (inputImage.getHeight() - 1);
    IntImage outputImage = IntImage(inputImage.getWidth(), inputImage.getHeight());
    uint32_t zero = 0;

    // Create Device Memory
    int32_t *d_inputImage;
    CHECK(cudaMalloc(&d_inputImage, inputImage.getWidth() * inputImage.getHeight() * sizeof(int32_t)))
    bool *isBlockFinished;
    CHECK(cudaMalloc(&isBlockFinished, gridSize * sizeof(bool)))

    // Copy Memory from Host to Device
    CHECK(cudaMemcpy(d_inputImage, inputImage.getPixels(),
                     inputImage.getWidth() * inputImage.getHeight() * sizeof(int32_t), cudaMemcpyHostToDevice))
    CHECK(cudaMemcpyToSymbol(KernelFunction::blockCount, &zero, sizeof(u_int32_t), 0, cudaMemcpyHostToDevice))
    CHECK(cudaMemset(isBlockFinished, 0, gridSize * sizeof(bool)))

    // Run Device Methods
    KernelFunction::updateSeamMapKernelPipelining<<<gridSize, blockSize>>>(d_inputImage, inputImage.getWidth(), isBlockFinished);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    // Copy Memory from Device to Host
    CHECK(cudaMemcpy(outputImage.getPixels(), d_inputImage,
                     outputImage.getWidth() * outputImage.getHeight() * sizeof(int32_t), cudaMemcpyDeviceToHost))

    // Free Device Memory
    CHECK(cudaFree(d_inputImage))
    CHECK(cudaFree(isBlockFinished))

    // Free Host Memory

    // Return result
    return outputImage;
}
