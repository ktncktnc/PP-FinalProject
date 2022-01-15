//
// Created by phuc on 12/01/2022.
//

#ifndef FINALPROJECT_PARALLEL_SOLUTION_V2_CUH
#define FINALPROJECT_PARALLEL_SOLUTION_V2_CUH

#include "parallel_solution_baseline.cuh"

class ParallelSolutionV2 : public ParallelSolutionBaseline{
    static IntImage calculateEnergyMap(const IntImage &inputImage, dim3 blockSize = dim3(32, 32));
    PnmImage run(const PnmImage &inputImage, int argc, char **argv);
};

namespace KernelFunction
{
    __global__ void
    convolutionKernel_v2(const int32_t *input, u_int32_t inputWidth, u_int32_t inputHeight, bool isXDirection,
                      u_int32_t filterSize, int32_t *output);

}


#endif //FINALPROJECT_PARALLEL_SOLUTION_V2_CUH
