#pragma once

#include "solution.cuh"

#define DEBUG_IDX = -1;

namespace SequentialFunction{
    void convolution(int *input, int inputWidth, int inputHeight, const int *filter, int filterSize, int* output);
    void convertToGray(uchar3* input, int width, int height, int* output);

    void addAbs(int *input_1, int* input_2, int inputWidth, int inputHeight, int *output);

    void createCumulativeEnergyMap(int* input, int inputWidth, int inputHeight, int* output);

    void findSeamCurve(int* input, int inputWidth, int inputHeight, int* output);

    int findMinIndex(int* arr, int size);

    void copyARow(int* input, int width, int height, int rowIdx, int removedIdx, int* output);

    void reduce(const uchar3* input, int width, int height, int* path, uchar3* output);
}

class SequentialSolution :public BaseSolution{
private:
    static const u_int32_t FILTER_SIZE = 3;
    static const int32_t SOBEL_X[9];
    static const int32_t SOBEL_Y[9];

public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;

    IntImage scan(const PnmImage &inputImage);

    void scan(uchar3* input, int width, int height, uchar3* output, int counter);
};
