#pragma once

#include "solution.cuh"

namespace SequentialFunction {
    void convolution(int *input, uint32_t inputWidth, uint32_t inputHeight, const int *filter, uint32_t filterSize, int *output);

    void convertToGray(uchar3 *input, uint32_t width, uint32_t height, int *output);

    void addAbs(int *input_1, int *input_2, uint32_t inputWidth, uint32_t inputHeight, int *output);

    void createCumulativeEnergyMap(int *input, uint32_t inputWidth, uint32_t inputHeight, int *output);

    void findSeamCurve(int *input, uint32_t inputWidth, uint32_t inputHeight, uint32_t *output);

    int findMinIndex(int *arr, uint32_t size);

    void copyARow(int *input, uint32_t width, uint32_t height, uint32_t rowIdx, int32_t removedIdx, int *output);

    void copyARow(uchar3 *input, uint32_t width, uint32_t height, uint32_t rowIdx, int32_t removedIdx, uchar3 *output);

    void reduce(uchar3 *input, uint32_t width, uint32_t height, uint32_t *path, uchar3 *output);
}

class SequentialSolution : public BaseSolution {
private:
    static const int FILTER_SIZE = 3;
    static const int SOBEL_X[9];
    static const int SOBEL_Y[9];

protected:
    static IntImage calculateSeamMap(const IntImage &inputImage);

    static IntImage convertToGrayScale(const PnmImage &inputImage);

    static IntImage calculateEnergyMap(const IntImage &inputImage);

    static PnmImage deleteSeam(const PnmImage &inputImage, uint32_t *seam);

    static void extractSeam(const IntImage &energyMap, uint32_t *seam);

public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;

};
