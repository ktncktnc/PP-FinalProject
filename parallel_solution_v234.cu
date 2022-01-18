#include "parallel_solution_v234.cuh"
#include "parallel_solution_v2.cuh"
#include "timer.cuh"

void ParallelSolutionV234::calculateEnergyMap(const int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight,
                                              dim3 blockSize, int32_t *d_outputImage) {
    // Create Host Memory
    dim3 gridSize((inputWidth - 1) / blockSize.x + 1, (inputHeight - 1) / blockSize.y + 1);
    size_t smemSize = (blockSize.x + FILTER_SIZE - 1) * (blockSize.y + FILTER_SIZE - 1) * sizeof(int32_t);

    // Create Device Memory
    int32_t *d_outputImageX;
    CHECK(cudaMalloc(&d_outputImageX, inputWidth * inputHeight * sizeof(int32_t)))
    int32_t *d_outputImageY;
    CHECK(cudaMalloc(&d_outputImageY, inputWidth * inputHeight * sizeof(int32_t)))

    // Copy Memory from Host to Device

    // Run Device Methods
    KernelFunction::convolutionKernel_v2<<<gridSize, blockSize, smemSize>>>(d_inputImage, inputWidth, inputHeight, true, FILTER_SIZE, d_outputImageX);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    KernelFunction::convolutionKernel_v2<<<gridSize, blockSize, smemSize>>>(d_inputImage, inputWidth, inputHeight, false, FILTER_SIZE, d_outputImageY);
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

PnmImage ParallelSolutionV234::run(const PnmImage &inputImage, int argc, char **argv) {

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
    printf("Running Parallel Solution Version 2 + 3 + 4 with blockSize=(%d;%d).\n", blockSize.x, blockSize.y);
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

    // Copy Memory from Host to Device
    CHECK(cudaMemcpy(d_inputImage, inputImage.getPixels(),
                     inputImage.getWidth() * inputImage.getHeight() * sizeof(uchar3), cudaMemcpyHostToDevice))

    CHECK(cudaMemcpyToSymbol(KernelFunction::c_filterX, SOBEL_X, FILTER_SIZE * FILTER_SIZE * sizeof(int32_t)))
    CHECK(cudaMemcpyToSymbol(KernelFunction::c_filterY, SOBEL_Y, FILTER_SIZE * FILTER_SIZE * sizeof(int32_t)))

    // Run Kernel functions
    convertToGrayScale(d_inputImage, inputImage.getWidth(), inputImage.getHeight(), blockSize, d_grayImage);
    for (int i = 0; i < nDeletingSeams; ++i) {
        // 1. Calculate the Energy Map
        stepTimer.Start();
        calculateEnergyMap(d_grayImage, inputImage.getWidth() - i, inputImage.getHeight(), blockSize, d_energyMap);
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

    // Free Host Memory
    free(seam);
    free(energyMap);

    // Stop Timer
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
    printf("Step time: 1) %.3f ms \t 2) %.3f ms \t 3) %.3f ms \t 4) %.3f ms\n", cal_energy_time, cal_seam_time, extract_seam_time, delete_seam_time);
    printf("-------------------------------\n");

    // Return
    return outputImage;
}
