#include "parallel_solution_v23.cuh"
#include "parallel_solution_v3.cuh"
#include "timer.cuh"

void ParallelSolutionV23::calculateSeamMap(int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight,
                                           uint32_t blockSize) {
    // Create Host Memory
    uint32_t gridSize = (inputWidth - 1) / blockSize + 1;

    // Create Device Memory

    // Run Device Methods
    cudaStream_t streamForward, streamBackward;
    cudaStreamCreate(&streamForward);
    cudaStreamCreate(&streamBackward);

    for (int i = 1; i < inputHeight / 2; ++i) {
        // Forward
        KernelFunction::updateSeamMapKernel<<<gridSize, blockSize, 0, streamForward>>>(d_inputImage, inputWidth, i);
        // Backward
        if (int(inputHeight) - i - 1 >= inputHeight / 2) {
            KernelFunction::updateSeamMapKernelBackward<<<gridSize, blockSize, 0, streamBackward>>>(d_inputImage, inputWidth,
                    int(inputHeight) - i - 1);
        }
        cudaStreamSynchronize(streamForward);
        cudaStreamSynchronize(streamBackward);
        CHECK(cudaGetLastError())
    }

    if (inputHeight % 2 == 1) {
        KernelFunction::updateSeamMapKernelBackward<<<gridSize, blockSize, 0, streamBackward>>>(d_inputImage, inputWidth,
                int(inputHeight) - int(inputHeight) / 2 - 1);
        cudaStreamSynchronize(streamForward);
        cudaStreamSynchronize(streamBackward);
        CHECK(cudaGetLastError())
    }

    cudaStreamDestroy(streamForward);
    cudaStreamDestroy(streamBackward);

    // Copy Memory from Device to Host

    // Free Device Memory

    // Free Host Memory

    // Return result
}

void
ParallelSolutionV23::extractSeam(const int32_t *energyMap, uint32_t inputWidth, uint32_t inputHeight, uint32_t *seam) {
    // Find minSeam
    u_int32_t minValCol1 = 0;
    u_int32_t minValCol2 = 0;
    u_int32_t middleRow = inputHeight / 2 - 1;
    int32_t bestVal = energyMap[KernelFunction::convertIndex(middleRow, 0, inputWidth)] +
                      energyMap[KernelFunction::convertIndex(middleRow + 1, 0, inputWidth)];

    for (int c = 0; c < inputWidth; ++c) {
        if (energyMap[KernelFunction::convertIndex(middleRow, c, inputWidth)] +
            energyMap[KernelFunction::convertIndex(middleRow + 1, c, inputWidth)]
            < bestVal) {
            bestVal = energyMap[KernelFunction::convertIndex(middleRow, c, inputWidth)] +
                      energyMap[KernelFunction::convertIndex(middleRow + 1, c, inputWidth)];
            minValCol1 = c;
            minValCol2 = c;
        }

        if (c > 0 &&
            energyMap[KernelFunction::convertIndex(middleRow, c - 1, inputWidth)] +
            energyMap[KernelFunction::convertIndex(middleRow + 1, c, inputWidth)]
            <= bestVal) {
            bestVal = energyMap[KernelFunction::convertIndex(middleRow, c - 1, inputWidth)] +
                      energyMap[KernelFunction::convertIndex(middleRow + 1, c, inputWidth)];
            minValCol1 = c - 1;
            minValCol2 = c;
        }

        if (c + 1 < inputWidth &&
            energyMap[KernelFunction::convertIndex(middleRow, c + 1, inputWidth)] +
            energyMap[KernelFunction::convertIndex(middleRow + 1, c, inputWidth)]
            < bestVal) {
            bestVal = energyMap[KernelFunction::convertIndex(middleRow, c + 1, inputWidth)] +
                      energyMap[KernelFunction::convertIndex(middleRow + 1, c, inputWidth)];
            minValCol1 = c + 1;
            minValCol2 = c;
        }
    }
    // Trace back
    seam[inputHeight / 2 - 1] = minValCol1;
    seam[inputHeight / 2] = minValCol2;

    for (int r = int(inputHeight / 2) - 2; r >= 0; --r) {
        auto c = minValCol1;
        if (c > 0) {
            if (energyMap[KernelFunction::convertIndex(r, c - 1, inputWidth)] <=
                energyMap[KernelFunction::convertIndex(r, minValCol1, inputWidth)]) {
                minValCol1 = c - 1;
            }
        }
        if (c + 1 < inputWidth) {
            if (energyMap[KernelFunction::convertIndex(r, c + 1, inputWidth)] <
                energyMap[KernelFunction::convertIndex(r, minValCol1, inputWidth)]) {
                minValCol1 = c + 1;
            }
        }
        seam[r] = minValCol1;
    }

    for (int r = int(inputHeight / 2) + 1; r < inputHeight; ++r) {
        auto c = minValCol2;
        if (c > 0) {
            if (energyMap[KernelFunction::convertIndex(r, c - 1, inputWidth)] <=
                energyMap[KernelFunction::convertIndex(r, minValCol2, inputWidth)]) {
                minValCol2 = c - 1;
            }
        }
        if (c + 1 < inputWidth) {
            if (energyMap[KernelFunction::convertIndex(r, c + 1, inputWidth)] <
                energyMap[KernelFunction::convertIndex(r, minValCol2, inputWidth)]) {
                minValCol2 = c + 1;
            }
        }
        seam[r] = minValCol2;
    }
}

PnmImage ParallelSolutionV23::run(const PnmImage &inputImage, int argc, char **argv) {
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
    printf("Running Parallel Solution Version 2 + 3 with blockSize=(%d;%d).\n", blockSize.x, blockSize.y);
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
    printf("Step time: 2) %.3f ms \t 3) %.3f ms \t 4) %.3f ms \t 5) %.3f ms\n", cal_energy_time, cal_seam_time, extract_seam_time, delete_seam_time);
    printf("-------------------------------\n");

    // Return
    return outputImage;
}
