#include "image.cuh"
#include "utils.cuh"
#include "solution.cuh"
#include "sequential_solution.cuh"
#include "parallel_solution_baseline.cuh"
#include "parallel_solution_v2.cuh"
#include "parallel_solution_v3.cuh"
#include "parallel_solution_v4.cuh"
#include "parallel_solution_v34.cuh"
#include "parallel_solution_v23.cuh"
#include "parallel_solution_v24.cuh"
#include "parallel_solution_v234.cuh"
#include <vector>

void printUserGuide(int list){
    if (list == 0)
        printf("Usage:\tFinalProject <input path> <output path> <solutionID> <n seam> (<blockSize.x> <blockSize.y>)\n");
    else{
        printf("Solution ID: \tnumber from 0 to 8\n"
               "\t\t0: all\n"
               "\t\t1: baseline solution\n"
               "\t\t2: solution 1\n"
               "\t\t3: solution 2\n"
               "\t\t4: solution 3\n"
               "\t\t5: solution 2 and 3\n"
               "\t\t6: solution 2 and 4\n"
               "\t\t7: solution 3 and 4\n"
               "\t\t8: solution 2, 3 and 4\n");
    }
}

bool extractFilesArgument(int argc, char **argv, char *&inputFileName, char *&outputFileName, int &solutionID) {
    if (argc < 4) {
        printf("The number of arguments is invalid\n");
        printUserGuide(0);
        return false;
    }
    inputFileName = argv[1];
    outputFileName = argv[2];
    solutionID = int(strtol(argv[3], nullptr, 10));
    return true;
}

const int N_PARALLEL_SOLUTIONS = 8;
BaseSolution *parallelSolutions[N_PARALLEL_SOLUTIONS];

int main(int argc, char **argv) {
    char *inputFilename, *outputFilename;
    int solutionID;
    PnmImage inputImage = PnmImage();

    printDeviceInfo();

    if (!extractFilesArgument(argc, argv, inputFilename, outputFilename, solutionID))
        return EXIT_FAILURE;

    if (solutionID < 0 || solutionID > N_PARALLEL_SOLUTIONS) {
        printf("The solution ID is invalid\n");
        printUserGuide(1);
        return EXIT_FAILURE;
    }

    inputImage.read(inputFilename);

    BaseSolution *sequentialSolution = new SequentialSolution();
    parallelSolutions[0] = new ParallelSolutionBaseline();
    parallelSolutions[1] = new ParallelSolutionV2();
    parallelSolutions[2] = new ParallelSolutionV3();
    parallelSolutions[3] = new ParallelSolutionV4();
    parallelSolutions[4] = new ParallelSolutionV23();
    parallelSolutions[5] = new ParallelSolutionV24();
    parallelSolutions[6] = new ParallelSolutionV34();
    parallelSolutions[7] = new ParallelSolutionV234();

    PnmImage outputImageSequential = sequentialSolution->run(inputImage, argc - 4, argv + 4);

    if (solutionID == 0) {
        // Print all Results
        printf("Running all Parallels solutions and compare outputs...\n");
        std::vector<PnmImage> outputImages;
        for (auto &parallelSolution: parallelSolutions)
            outputImages.push_back(parallelSolution->run(inputImage, argc - 4, argv + 4));
        for (auto &outputImage: outputImages)
            outputImageSequential.compare(outputImage);
    } else {
        printf("Running one solution...\n");
        PnmImage outputParallel = parallelSolutions[solutionID - 1]->run(inputImage, argc - 4, argv + 4);
        outputImageSequential.compare(outputParallel);
        printf(R"(Outputs are written to "sequential_solution.pnm" (Sequential Output) and "%s" (Parallel Output).)",
               outputFilename);
        outputImageSequential.write("sequential_solution.pnm");
        outputParallel.write(outputFilename);
    }

    free(sequentialSolution);
    for (auto &parallelSolution: parallelSolutions)
        free(parallelSolution);

    return 0;
}
