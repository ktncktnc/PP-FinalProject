#ifndef FINALPROJECT_PARALLEL_SOLUTION_V2_CUH
#define FINALPROJECT_PARALLEL_SOLUTION_V2_CUH

#include "parallel_solution_baseline.cuh"

class ParallelSolutionV2 : public ParallelSolutionBaseline {
private:
    static void calculateEnergyMap(const int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight,
                                   dim3 blockSize, int32_t *d_outputImage);
public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;
};

namespace KernelFunction {
    __global__ void
    convolutionKernel_v2(const int32_t *input, u_int32_t inputWidth, u_int32_t inputHeight, bool isXDirection,
                         u_int32_t filterSize, int32_t *output);

}


#endif //FINALPROJECT_PARALLEL_SOLUTION_V2_CUH
