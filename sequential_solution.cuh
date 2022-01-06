#pragma once

#include "solution.cuh"

class SequentialSolution :public BaseSolution{
public:
    PnmImage run(const PnmImage &inputImage, char **argv) override;
};
