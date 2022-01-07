#include "parallel_solution_baseline.cuh"

PnmImage ParallelSolutionBaseline::run(const PnmImage &inputImage, int argc, char **argv) {
    // 1. Scan
    PnmImage scannedImage = this->scan(inputImage);
    return BaseSolution::run(inputImage, argc, argv);
}

PnmImage ParallelSolutionBaseline::scan(const PnmImage &inputImage) {
    PnmImage outputImage = PnmImage(inputImage.getWidth(), inputImage.getHeight());
    return outputImage;
}
