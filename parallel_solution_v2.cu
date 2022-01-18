#include "timer.cuh"
#include "parallel_solution_v2.cuh"

namespace KernelFunction {
    __constant__ int32_t c_filterX[TOTAL_FILTER_SIZE];
    __constant__ int32_t c_filterY[TOTAL_FILTER_SIZE];

    __global__ void
    convolutionKernel_v2(const int32_t *input, u_int32_t width, u_int32_t height, bool isXDirection,
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

                int32_t filterVal;
                if (isXDirection) {
                    filterVal = c_filterX[convertIndex(k_r + filterSize / 2, k_c + filterSize / 2, filterSize)];
                } else {
                    filterVal = c_filterY[convertIndex(k_r + filterSize / 2, k_c + filterSize / 2, filterSize)];
                }
                outPixel += inPixel * filterVal;
            }
        }
        output[out_idx] = outPixel;
    }
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

    // Start Timer
    printf("Running Parallel Solution Version 2 with blockSize=(%d;%d).\n", blockSize.x, blockSize.y);
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

    CHECK(cudaMemcpyToSymbol(KernelFunction::c_filterX, SOBEL_X, FILTER_SIZE * FILTER_SIZE * sizeof(int32_t)))
    CHECK(cudaMemcpyToSymbol(KernelFunction::c_filterY, SOBEL_Y, FILTER_SIZE * FILTER_SIZE * sizeof(int32_t)))

    // Copy Memory from Host to Device
    CHECK(cudaMemcpy(d_inputImage, inputImage.getPixels(),
                     inputImage.getWidth() * inputImage.getHeight() * sizeof(uchar3), cudaMemcpyHostToDevice))

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
    printf("Step time: 1/%.3f ms 2/%.3f ms 3/%.3f ms 4/%.3f ms", cal_energy_time, cal_seam_time, extract_seam_time, delete_seam_time);
    printf("-------------------------------\n");

    // Return
    return outputImage;
}

void
ParallelSolutionV2::calculateEnergyMap(const int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight,
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
