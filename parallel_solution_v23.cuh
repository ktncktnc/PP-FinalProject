//
// Created by phuc on 16/01/2022.
//

#ifndef FINALPROJECT_PARALLEL_SOLUTION_V23_CUH
#define FINALPROJECT_PARALLEL_SOLUTION_V23_CUH


#include "parallel_solution_baseline.cuh"
#include "parallel_solution_v2.cuh"

class ParallelSolutionV23 : public ParallelSolutionV2 {
private:
    static void calculateSeamMap(int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight, uint32_t blockSize);

    static void extractSeam(const int32_t *energyMap, uint32_t inputWidth, uint32_t inputHeight, uint32_t *seam);

public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;
};


#endif //FINALPROJECT_PARALLEL_SOLUTION_V23_CUH
