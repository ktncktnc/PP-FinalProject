#ifndef FINALPROJECT_PARALLEL_SOLUTION_V34_CUH
#define FINALPROJECT_PARALLEL_SOLUTION_V34_CUH


#include "parallel_solution_baseline.cuh"

namespace KernelFunction {
    extern __device__ u_int32_t blockCountBackward;
    extern __device__ u_int32_t blockCountForward;

    __global__ void updateSeamMapKernelPipeliningBackward(int32_t *input, u_int32_t inputWidth, u_int32_t inputHeight,
                                                          bool volatile *isBlockFinished);

    __global__ void
    updateSeamMapKernelPipeliningForward(int32_t *input, u_int32_t inputWidth,
                                         bool volatile *isBlockFinished);
}

class ParallelSolutionV34 : public ParallelSolutionBaseline {

protected:
    static void calculateSeamMap(int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight, uint32_t blockSize);

    static void extractSeam(const int32_t *energyMap, uint32_t inputWidth, uint32_t inputHeight, uint32_t *seam);

public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;
};


#endif //FINALPROJECT_PARALLEL_SOLUTION_V34_CUH
