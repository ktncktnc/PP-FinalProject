//
// Created by phuc on 12/01/2022.
//

#ifndef FINALPROJECT_PARALLEL_SOLUTION_V3_CUH
#define FINALPROJECT_PARALLEL_SOLUTION_V3_CUH


#include "parallel_solution_baseline.cuh"

namespace KernelFunction {
    __global__ void
    updateSeamMapKernelV2(int32_t *input, u_int32_t inputWidth,
                          int32_t currentRow);
}

class ParallelSolutionV3 : public ParallelSolutionBaseline {
private:
    static IntImage calculateSeamMap(const IntImage &inputImage, uint32_t blockSize = 32);
public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;
};


#endif //FINALPROJECT_PARALLEL_SOLUTION_V3_CUH
