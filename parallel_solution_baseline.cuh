#pragma once

#include "solution.cuh"

class ParallelSolutionBaseline : public BaseSolution {
public:
    PnmImage run(const PnmImage &inputImage, char **argv) override;
};
