#include "parallel_solution_v3.cuh"
#include "timer.cuh"

namespace KernelFunction {
    __global__ void
    updateSeamMapKernelV2(int32_t *input, u_int32_t inputWidth,
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
}

IntImage ParallelSolutionV3::calculateSeamMap(const IntImage &inputImage, uint32_t blockSize) {
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
        KernelFunction::updateSeamMapKernelV2<<<gridSize, blockSize>>>(d_inputImage, inputImage.getWidth(), i);
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

PnmImage ParallelSolutionV3::run(const PnmImage &inputImage, int argc, char **argv) {
    // Extract arguments
    int nDeletingSeams = 1;
    dim3 blockSize(32, 32); // Default
    if (argc > 0)
        nDeletingSeams = int(strtol(argv[0], nullptr, 10));
    if (argc > 1) {
        blockSize.x = strtol(argv[1], nullptr, 10);
        blockSize.y = strtol(argv[2], nullptr, 10);
    }
    printf("Running Parallel Solution Version 3 with blockSize=(%d;%d).\n", blockSize.x, blockSize.y);
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
