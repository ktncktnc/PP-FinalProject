#pragma once

#include "solution.cuh"

class SequentialSolution :public BaseSolution{
private:
    static const u_int32_t FILTER_SIZE = 3;
    static const int32_t SOBEL_X[3][3];
    static const int32_t SOBEL_Y[3][3];

public:
    IntImage scan(const PnmImage &inputImage);
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;
};
