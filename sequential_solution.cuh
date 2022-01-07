#pragma once

#include "solution.cuh"

class SequentialSolution :public BaseSolution{
public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;
};
