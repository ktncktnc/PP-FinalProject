#pragma once

#include "solution.cuh"

namespace SequentialFunction {
    void convolution(int *input, int inputWidth, int inputHeight, const int *filter, int filterSize, int *output);

    void convertToGray(uchar3 *input, int width, int height, int *output);

    void addAbs(int *input_1, int *input_2, int inputWidth, int inputHeight, int *output);

    void createCumulativeEnergyMap(int *input, int inputWidth, int inputHeight, int *output);

    void findSeamCurve(int *input, int inputWidth, uint32_t int, uint32_t *output);

    int findMinIndex(int *arr, int size);

    void copyARow(int *input, uint32_t width, int height, int rowIdx, int removedIdx, int *output);

    void copyARow(uchar3 *input, int width, int height, int rowIdx, int removedIdx, uchar3 *output);

    void reduce(uchar3 *input, int width, int height, uint32_t *path, uchar3 *output);
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
