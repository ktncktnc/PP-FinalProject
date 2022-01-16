#ifndef FINALPROJECT_PARALLEL_SOLUTION_V24_CUH
#define FINALPROJECT_PARALLEL_SOLUTION_V24_CUH


#include "parallel_solution_v2.cuh"

class ParallelSolutionV24 : public ParallelSolutionV2 {
private:
    static void calculateSeamMap(int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight, uint32_t blockSize);

public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;
};


#endif //FINALPROJECT_PARALLEL_SOLUTION_V24_CUH
