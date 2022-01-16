#include "image.cuh"
#include "utils.cuh"
#include "solution.cuh"
#include "sequential_solution.cuh"
#include "parallel_solution_baseline.cuh"
#include "parallel_solution_v2.cuh"
#include "parallel_solution_v3.cuh"
#include "parallel_solution_v4.cuh"

bool extractFilesArgument(int argc, char **argv, char *&inputFileName, char *&outputFileName, int &solutionID) {
    if (argc < 4) {
        printf("The number of arguments is invalid\n");
        return false;
    }
    inputFileName = argv[1];
    outputFileName = argv[2];
    solutionID = int(strtol(argv[3], nullptr, 10));
    return true;
}

const int N_PARALLEL_SOLUTIONS = 4;
BaseSolution *parallelSolutions[N_PARALLEL_SOLUTIONS];

int main(int argc, char **argv) {
    char *inputFilename, *outputFilename;
    int solutionID;
    PnmImage inputImage = PnmImage();

    printDeviceInfo();

    if (!extractFilesArgument(argc, argv, inputFilename, outputFilename, solutionID))
        return EXIT_FAILURE;

    if (solutionID < 1 || solutionID > N_PARALLEL_SOLUTIONS) {
        printf("The solution ID is invalid\n");
        return EXIT_FAILURE;
    }

    inputImage.read(inputFilename);

    BaseSolution *sequentialSolution = new SequentialSolution();
    parallelSolutions[0] = new ParallelSolutionBaseline();
    parallelSolutions[1] = new ParallelSolutionV2();
    parallelSolutions[2] = new ParallelSolutionV3();
    parallelSolutions[3] = new ParallelSolutionV4();

    PnmImage outputImageSequential = sequentialSolution->run(inputImage, argc - 4, argv + 4);
    PnmImage outputImageParallel = parallelSolutions[solutionID - 1]->run(inputImage, argc - 4, argv + 4);


    outputImageSequential.write("sequential_solution.pnm");
    outputImageParallel.write(outputFilename);

    outputImageSequential.compare(outputImageParallel);

    free(sequentialSolution);
    free(parallelSolutions[0]);
    free(parallelSolutions[1]);
    free(parallelSolutions[2]);
    free(parallelSolutions[3]);

    return 0;
}
