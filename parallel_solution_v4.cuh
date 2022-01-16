#ifndef FINALPROJECT_PARALLEL_SOLUTION_V4_CUH
#define FINALPROJECT_PARALLEL_SOLUTION_V4_CUH


#include "parallel_solution_baseline.cuh"

namespace KernelFunction {
    __global__ void
    updateSeamMapKernelPipelining(int32_t *input, u_int32_t inputWidth,
                                  bool volatile *isBlockFinished);
}
class ParallelSolutionV4 : public ParallelSolutionBaseline {
private:
    static IntImage calculateSeamMap(const IntImage &inputImage, uint32_t blockSize = 32);

public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;
};


#endif //FINALPROJECT_PARALLEL_SOLUTION_V4_CUH
