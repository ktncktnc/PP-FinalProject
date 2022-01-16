//
// Created by phuc on 16/01/2022.
//

#include "parallel_solution_v24.cuh"
#include "timer.cuh"
#include "parallel_solution_v4.cuh"

void ParallelSolutionV24::calculateSeamMap(int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight,
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

PnmImage ParallelSolutionV24::run(const PnmImage &inputImage, int argc, char **argv) {
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
    printf("Running Parallel Solution Version 2 + 4 with blockSize=(%d;%d).\n", blockSize.x, blockSize.y);
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

    // Copy Memory from Host to Device
    CHECK(cudaMemcpy(d_inputImage, inputImage.getPixels(),
                     inputImage.getWidth() * inputImage.getHeight() * sizeof(uchar3), cudaMemcpyHostToDevice))

    CHECK(cudaMemcpyToSymbol(KernelFunction::c_filterX, SOBEL_X, FILTER_SIZE * FILTER_SIZE * sizeof(int32_t)))
    CHECK(cudaMemcpyToSymbol(KernelFunction::c_filterY, SOBEL_Y, FILTER_SIZE * FILTER_SIZE * sizeof(int32_t)))

    // Run Kernel functions
    convertToGrayScale(d_inputImage, inputImage.getWidth(), inputImage.getHeight(), blockSize, d_grayImage);
    for (int i = 0; i < nDeletingSeams; ++i) {
        // 1. Calculate the Energy Map
        calculateEnergyMap(d_grayImage, inputImage.getWidth() - i, inputImage.getHeight(), blockSize, d_energyMap);
        // 2. Dynamic Programming
        calculateSeamMap(d_energyMap, inputImage.getWidth() - i, inputImage.getHeight(), blockSize.x * blockSize.y);
        // 3. Extract the seam
        CHECK(cudaMemcpy(energyMap, d_energyMap,
                         (inputImage.getWidth() - i) * inputImage.getHeight() * sizeof(int32_t),
                         cudaMemcpyDeviceToHost))
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
