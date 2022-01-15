#pragma once

#include "solution.cuh"

namespace SequentialFunction {
    void convolution(int32_t *input, uint32_t inputWidth, uint32_t inputHeight, const int32_t *filter, uint32_t filterSize, int32_t *output);

    void convertToGray(uchar3 *input, uint32_t width, uint32_t height, int32_t *output);

    void addAbs(int32_t *input_1, int32_t *input_2, uint32_t inputWidth, uint32_t inputHeight, int32_t *output);

    void createCumulativeEnergyMap(int32_t *input, uint32_t inputWidth, uint32_t inputHeight, int32_t *output);

    void findSeamCurve(int32_t *input, uint32_t inputWidth, uint32_t inputHeight, uint32_t *output);

    int findMinIndex(int32_t *arr, uint32_t size);

    void copyARow(int32_t *input, uint32_t width, int rowIdx, int32_t *output);

    void copyARowAndRemove(uchar3 *input, uint32_t width, int rowIdx, int removedIdx, uchar3 *output);

    void reduce(uchar3 *input, uint32_t width, uint32_t height, uint32_t *path, uchar3 *output);
}

class SequentialSolution : public BaseSolution {
private:
    static const uint32_t FILTER_SIZE = 3;
    static const int32_t SOBEL_X[9];
    static const int32_t SOBEL_Y[9];

protected:
    static IntImage calculateSeamMap(const IntImage &inputImage);

    static IntImage convertToGrayScale(const PnmImage &inputImage);

    static IntImage calculateEnergyMap(const IntImage &inputImage);

    static PnmImage deleteSeam(const PnmImage &inputImage, uint32_t *seam);

    static void extractSeam(const IntImage &energyMap, uint32_t *seam);

public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;

};
