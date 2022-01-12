#include "image.cuh"
#include "utils.cuh"
#include "solution.cuh"
#include "sequential_solution.cuh"
#include "parallel_solution_baseline.cuh"
#include "parallel_solution_v3.cuh"

bool extractFilesArgument(int argc, char **argv, char *&inputFileName) {
    if (argc < 2) {
        printf("The number of arguments is invalid\n");
        return false;
    }
    inputFileName = argv[1];
    return true;
}

int main(int argc, char **argv) {
    char *inputFilename;
    PnmImage inputImage = PnmImage();

    printDeviceInfo();

    if (!extractFilesArgument(argc, argv, inputFilename))
        return EXIT_FAILURE;
    inputImage.read(inputFilename);

    BaseSolution *sequentialSolution = new SequentialSolution();
    BaseSolution *parallelSolution = new ParallelSolutionBaseline();
    BaseSolution *parallelSolutionV3 = new ParallelSolutionV3();

    PnmImage outputImageSequential = sequentialSolution->run(inputImage, argc - 2, argv + 2);
    PnmImage outputImageParallel = parallelSolution->run(inputImage, argc - 2, argv + 2);
    PnmImage outputImageParallelV3 = parallelSolutionV3->run(inputImage, argc - 2, argv + 2);

    outputImageSequential.write("output_sequential.pnm");
    outputImageParallel.write("output_parallel.pnm");
    outputImageParallelV3.write("output_parallelV3.pnm");

    outputImageSequential.compare(outputImageParallel);
    outputImageSequential.compare(outputImageParallelV3);

    free(sequentialSolution);
    free(parallelSolution);
    free(parallelSolutionV3);
    
    return 0;
}
