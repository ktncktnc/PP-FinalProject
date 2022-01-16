//
// Created by phuc on 16/01/2022.
//

#ifndef FINALPROJECT_PARALLEL_SOLUTION_V234_CUH
#define FINALPROJECT_PARALLEL_SOLUTION_V234_CUH


#include "parallel_solution_v34.cuh"

class ParallelSolutionV234 : public ParallelSolutionV34 {
    static void calculateEnergyMap(const int32_t *d_inputImage, uint32_t inputWidth, uint32_t inputHeight,
                                   dim3 blockSize, int32_t *d_outputImage);

public:
    PnmImage run(const PnmImage &inputImage, int argc, char **argv) override;
};


#endif //FINALPROJECT_PARALLEL_SOLUTION_V234_CUH
