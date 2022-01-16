//
// Created by phuc on 12/01/2022.
//

#ifndef FINALPROJECT_PARALLEL_SOLUTION_V3_CUH
#define FINALPROJECT_PARALLEL_SOLUTION_V3_CUH


#include "parallel_solution_baseline.cuh"

namespace KernelFunction {
    __global__ void
    updateSeamMapKernelBackward(int32_t *input, u_int32_t inputWidth,
                                int32_t currentRow);
}

class ParallelSolutionV3 : public ParallelSolutionBaseline {
private:
    static void calculateSeamMap(int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight, uint32_t blockSize);

    static void extractSeam(const int32_t *energyMap, uint32_t inputWidth, uint32_t inputHeight, uint32_t *seam);

public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;
};


#endif //FINALPROJECT_PARALLEL_SOLUTION_V3_CUH
