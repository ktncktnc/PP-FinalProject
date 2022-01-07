#pragma once

#include "solution.cuh"

class ParallelSolutionBaseline : public BaseSolution {
private:
    static const u_int32_t FILTER_WIDTH = 3;
    constexpr static const int32_t SOBEL_X[3][3] = {{1, 0, -1},
                                                    {2, 0, -2},
                                                    {1, 0, -1}};
    constexpr static const int32_t SOBEL_Y[3][3] = {{1,  2,  1},
                                                    {0,  0,  0},
                                                    {-1, -2, -1}};
public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;
    PnmImage scan(const PnmImage &inputImage);
};
