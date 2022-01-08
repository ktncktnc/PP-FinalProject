#pragma once

#include "solution.cuh"

namespace SequentialFunction{
    void scan(const int *input, u_int32_t inputWidth, u_int32_t inputHeight, const int3 *filter,
              u_int32_t filterSize, int* output);

    void convertToGray(PnmImage &inputImage, int* output);

    void addAbs(const int *input_1, const int* input_2, u_int32_t inputWidth, u_int32_t inputHeight,
                int *output);

    void createCumulativeEnergyMap(
            const int* input, //Gradient image
            u_int32_t inputWidth,
            u_int32_t inputHeight,
            bool direction //Direction: 0: vertical 1: horizontal,
            long* output
    );

    void findSeamCurve(
            const long* input,
            bool direction,
            u_int32_t inputWidth,
            u_int32_t inputHeight,
            int* output
    );

    int findMinIndex(int* arr, int size);
}

class SequentialSolution :public BaseSolution{
private:
    static const u_int32_t FILTER_SIZE = 3;
    static const int32_t SOBEL_X[3][3];
    static const int32_t SOBEL_Y[3][3];

public:
    IntImage scan(const PnmImage &inputImage);
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;
};
