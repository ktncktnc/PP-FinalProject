#include "sequential_solution.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include <string>

using namespace std;

namespace SequentialFunction {
    // Convolution func
    void convolution(int32_t *input, uint32_t inputWidth, uint32_t inputHeight, const int32_t *filter, uint32_t filterSize, int32_t *output) {
        int index, k_index, k_x, k_y;
        int32_t sum;

        //For each pixel in image
        for (int x = 0; x < int(inputHeight); x++) {
            for (int y = 0; y < int(inputWidth); y++) {
                sum = 0;
                index = x * inputWidth + y;
                //For each value in kernel
                for (int i = -int(filterSize / 2); i <= int(filterSize / 2); i++) {
                    for (int j = -int(filterSize / 2); j <= int(filterSize / 2); j++) {
                        k_x = min(max(x + i, 0), int32_t(inputHeight) - 1);
                        k_y = min(max(y + j, 0), int32_t(inputWidth) - 1);
                        k_index = k_x * inputWidth + k_y;
                        sum += input[k_index] * filter[(i + int(filterSize / 2)) * int(filterSize) + j + int(filterSize / 2)];
                    }
                }
                output[index] = sum;
            }
        }
    }

    // Convert RGB to gray
    void convertToGray(uchar3 *input, uint32_t width, uint32_t height, int32_t *output) {
        for (int i = 0; i < width * height; i++) {
            output[i] = int32_t(299 * input[i].x + 587 * input[i].y + 114 * input[i].z) / 1000;
        }
    }

    // Create energy arr from X and Y
    void addAbs(int32_t *input_1, int32_t *input_2, uint32_t inputWidth, uint32_t inputHeight, int32_t *output) {
        int index;
        int32_t value;
        for (int x = 0; x < inputHeight; x++) {
            for (int y = 0; y < inputWidth; y++) {
                index = x * inputWidth + y;
                value = abs(input_1[index]) + abs(input_2[index]);
                output[index] = value;
            }
        }
    }

    // Create cumulative map
    void createCumulativeEnergyMap(int32_t *input, uint32_t inputWidth, uint32_t inputHeight, int32_t *output) {
        int a, b, c;
        // Copy last line
        copyARow(input, inputWidth, 0, output);

        for (int row = 1; row < inputHeight; row++) {
            for (int col = 0; col < inputWidth; col++) {
                a = output[(row - 1) * inputWidth + max(col - 1, 0)];
                b = output[(row - 1) * inputWidth + col];
                c = output[(row - 1) * inputWidth + min(col + 1, int32_t(inputWidth) - 1)];

                output[row * inputWidth + col] = input[row * inputWidth + col] + min(min(a, b), c);
            }
        }
    }

    // Find seam curve from cumulative map
    void findSeamCurve(int32_t *input, uint32_t inputWidth, uint32_t inputHeight, uint32_t *output) {
        int a, b, c, min_idx, offset, best;
        min_idx = findMinIndex(input + (int32_t(inputHeight) - 1) * inputWidth, inputWidth);
        output[int32_t(inputHeight) - 1] = min_idx;

        for (int row = int32_t(inputHeight) - 2; row >= 0; row--) {
            a = input[row * inputWidth + max(min_idx - 1, 0)];
            b = input[row * inputWidth + min_idx];
            c = input[row * inputWidth + min(min_idx + 1, int32_t(inputWidth) - 1)];
            offset = 0;
            best = b;
            if (a <= best) {
                best = a;
                offset = -1;
            }
            if (c < best) {
                offset = 1;
            }
            min_idx = min(max(min_idx + offset, 0), int32_t(inputWidth) - 1);
            output[row] = min_idx;
        }
    }

    // Remove seam curve from image
    void reduce(uchar3 *input, uint32_t width, uint32_t height, uint32_t *path, uchar3 *output) {
        for (int i = 0; i < height; i++) {
            copyARowAndRemove(input, width, i, int(path[i]), output);
        }
    }

    // Util funcs--------------------
    int findMinIndex(int32_t *arr, uint32_t size) {
        int min_idx = 0;
        for (int i = 1; i < size; i++) {
            if (arr[min_idx] > arr[i])
                min_idx = i;
        }
        return min_idx;
    }

    void copyARow(int32_t *input, uint32_t width, int32_t rowIdx, int32_t *output) {
        int output_idx = rowIdx * width, input_idx;

        for (int i = 0; i < width; i++) {
            input_idx = rowIdx * width + i;
            output[output_idx] = input[input_idx];
            output_idx++;
        }
    }

    void copyARowAndRemove(uchar3 *input, uint32_t width, int32_t rowIdx, int32_t removedIdx, uchar3 *output) {
        int output_idx = rowIdx * (width - 1), input_idx;
        for (int i = 0; i < width; i++) {
            if (i == removedIdx) continue;
            input_idx = rowIdx * width + i;
            output[output_idx] = input[input_idx];
            output_idx++;
        }
    }
}

const int SequentialSolution::SOBEL_X[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
const int SequentialSolution::SOBEL_Y[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

PnmImage SequentialSolution::run(const PnmImage &inputImage, int argc, char **argv) {
    if (argc < 1) {
        printf("The number of arguments is invalid\n");
        return PnmImage(inputImage.getWidth(), inputImage.getHeight());
    }
    int nDeletingSeams = stoi(argv[0], nullptr);

    printf("Running Baseline Sequential Solution\n");

    GpuTimer timer;
    timer.Start();

    PnmImage outputImage = inputImage;

    for (int i = 0; i < nDeletingSeams; ++i) {
        // 1. Convert to GrayScale
        IntImage grayImage = convertToGrayScale(outputImage);
        // 2. Calculate the Energy Map
        IntImage energyMap = calculateEnergyMap(grayImage);
        // 3. Dynamic Programming
        IntImage seamMap = calculateSeamMap(energyMap);
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

IntImage SequentialSolution::convertToGrayScale(const PnmImage &inputImage){
    IntImage outputImage = IntImage(inputImage.getWidth(), inputImage.getHeight());
    SequentialFunction::convertToGray(inputImage.getPixels(), inputImage.getWidth(), inputImage.getHeight(), outputImage.getPixels());

    return outputImage;
}

IntImage SequentialSolution::calculateEnergyMap(const IntImage &inputImage){
    uint32_t width = inputImage.getWidth(), height = inputImage.getHeight();

    IntImage gradX = IntImage(inputImage.getWidth(), inputImage.getHeight());
    IntImage gradY = IntImage(inputImage.getWidth(), inputImage.getHeight());
    IntImage grad = IntImage(inputImage.getWidth(), inputImage.getHeight());

    SequentialFunction::convolution(inputImage.getPixels(), width, height, SOBEL_X, FILTER_SIZE, gradX.getPixels());
    SequentialFunction::convolution(inputImage.getPixels(), width, height, SOBEL_Y, FILTER_SIZE, gradY.getPixels());

    // Cal energy
    SequentialFunction::addAbs(gradX.getPixels(), gradY.getPixels(), width, height, grad.getPixels());

    return grad;
}

IntImage SequentialSolution::calculateSeamMap(const IntImage &inputImage){
    IntImage map = IntImage(inputImage.getWidth(), inputImage.getHeight());
    SequentialFunction::createCumulativeEnergyMap(inputImage.getPixels(), inputImage.getWidth(), inputImage.getHeight(), map.getPixels());

    return map;
};

void SequentialSolution::extractSeam(const IntImage &energyMap, uint32_t *seam){
    SequentialFunction::findSeamCurve(energyMap.getPixels(), energyMap.getWidth(), energyMap.getHeight(), seam);
};

PnmImage SequentialSolution::deleteSeam(const PnmImage &inputImage, uint32_t *seam){
    PnmImage outputImage = PnmImage(inputImage.getWidth() - 1, inputImage.getHeight());

    SequentialFunction::reduce(inputImage.getPixels(), inputImage.getWidth(), inputImage.getHeight(), seam, outputImage.getPixels());
    return outputImage;
};

//uchar3* SequentialSolution::scan(uchar3 *input, int width, int height, int counter) {
//    int output_width = width - 1;
//    int output_height = height;
//
//    uchar3 *output = (uchar3 *) malloc(output_width * output_height * sizeof(uchar3));
//
//    // Convert to gray image
//    int *grayImg = (int *) malloc(width * height * sizeof(int));
//
//
//    // Convolution
//    int *gradX, *gradY, *grad;
//    gradX = (int *) malloc(width * height * sizeof(int));
//    gradY = (int *) malloc(width * height * sizeof(int));
//    grad = (int *) malloc(width * height * sizeof(int));
//
//    SequentialFunction::convolution(grayImg, width, height, SOBEL_X, FILTER_SIZE, gradX);
//    SequentialFunction::convolution(grayImg, width, height, SOBEL_Y, FILTER_SIZE, gradY);
//
//    // Cal energy
//    SequentialFunction::addAbs(gradX, gradY, width, height, grad);
//
//    // Cal cumulative map
//    int *map = (int *) malloc(width * height * sizeof(int));
//    SequentialFunction::createCumulativeEnergyMap(grad, width, height, map);
//
//    // Cal path
//    int *path = (int *) malloc(height * sizeof(int));
//    SequentialFunction::findSeamCurve(map, width, height, path);
//
//    // Remove seam curve
//    SequentialFunction::reduce(input, width, height, path, output);
//
//    free(grayImg);
//    free(gradX);
//    free(gradY);
//    free(grad);
//    free(map);
//    free(path);
//
//    return output;
//}