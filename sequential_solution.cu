#include "sequential_solution.cuh"
#include <iostream>

using namespace std;

namespace SequentialFunction {
    void scan(const int *input, u_int32_t inputWidth, u_int32_t inputHeight, const int3 *filter,
              u_int32_t filterSize, int* output) {
        int index, k_index, k_x, k_y, k_value, sum;

        //For each pixel in image
        for(int x = 0; x < inputHeight; x++){
            for(int y = 0; y < inputWidth; y++){
                sum = 0;
                index = x*inputWidth + y;

                //For each value in kernel
                for(int i = -(filterSize/2); i <= filterSize/2; i++){
                    for(int j = -(filterSize/2); j <= filterSize/2; j++){
                        k_x = min(max(x + i, 0), inputHeight - 1);
                        k_y = min(max(y + j, 0), inputWidth - 1);

                        k_index = k_x*inputWidth + k_y;

                        k_value = (j % 3 == 0)*filter[i].x + (j % 3 == 1)*filter[i].y + (j % 3 == 2)* filter[i].z;
                        sum += input[index] * value;
                    }
                }
                output[index] = sum;
            }
        }
    }

    void convertToGray(PnmImage &inputImage, int* output){
        uchar3 *input = inputImage.getPixels();

        for(int i = 0; i < inputImage.getWidth() * inputImage.getHeight(); i++){
            output[i] = int(0.299f*input[i].x) + int(0.587f*input[i].y) + int(0.114f*input[i].z);
        }
    }

    void addAbs(const int *input_1, const int* input_2, u_int32_t inputWidth, u_int32_t inputHeight,
                 int *output) {
        int index;

        for(int x = 0; x < inputHeight; x++){
            for(int y = 0; y < inputWidth; y++) {
                index = x*inputWidth + y;
                int value = sqrt(pow(input_1[index], 2) + pow(input_2[index], 2)));
                output[index] = value;
            }
        }
    }

    void createCumulativeEnergyMap(
            const int* input, //Gradient image
            u_int32_t inputWidth,
            u_int32_t inputHeight,
            bool direction //Direction: 0: vertical 1: horizontal,
            long* output
            ){

        int a, b, c;

        // Copy first line
        if (direction == 0){
            memcpy(output, input, inputWidth * sizeof(int));
        }
        else{
            for(int i = 0; i < inputHeight; i++){
                output[i* inputWidth] = input[i * inputWidth];
            }
        }

        if (direction == 0){
            for(int row = 1; row < inputHeight; row++){
                for(int col = 0; col < inputWidth; col++){
                    a = output[(row - 1)*inputWidth + max(col - 1, 0)];
                    b = output[(row - 1)*inputWidth + col];
                    c = output[(row - 1)*inputWidth + min(col + 1, inputWidth - 1)];

                    output[row*inputWidth + col] = min(min(a, b), c);
                }
            }
        }
        else
            for(int col = 1; col < inputWidth; col++){
                for(int row = 0; row < inputHeight; row++){
                    a = output[max(row - 1, 0)*inputWidth + col - 1];
                    b = output[row*inputWidth + col - 1];
                    c = output[min(row + 1, inputHeight - 1)*inputWidth + col - 1];

                    output[row*inputWidth + col] = min(min(a, b), c);
                }
            }
        }

    void findSeamCurve(
            const long* input,
            bool direction,
            u_int32_t inputWidth,
            u_int32_t inputHeight,
            int* output
            ){
        if (direction == 0){

        }
    }
}

const int32_t ParallelSolutionBaseline::SOBEL_X[3][3] = {{1, 0, -1},
                                                         {2, 0, -2},
                                                         {1, 0, -1}};
const int32_t ParallelSolutionBaseline::SOBEL_Y[3][3] = {{1,  2,  1},
                                                         {0,  0,  0},
                                                         {-1, -2, -1}};

PnmImage SequentialSolution::run(const PnmImage &inputImage, int argc, char **argv) {
    return BaseSolution::run(inputImage, argc, argv);
}

IntImage SequentialSolution::scan(const PnmImage &inputImage) {
    int* grayImg = (int*)malloc(inputImage.getHeight() * inputImage.getWidth());

    int* grImgX, *grImgY, *grImg;
    grImgX = (int*)malloc(inputImage.getHeight() * inputImage.getWidth());
    grImgY = (int*)malloc(inputImage.getHeight() * inputImage.getWidth());
    grImg = (int*)malloc(inputImage.getHeight() * inputImage.getWidth());;

    //RGB to gray
    SequentialFunction::toGray(inputImage, grayImg);

    //Scan
    SequentialFunction::scan(grayImg, inputImage.getWidth(), inputImage.getHeight(), SOBEL_X, 3,grImgX);
    SequentialFunction::scan(grayImg, inputImage.getWidth(), inputImage.getHeight(), SOBEL_X, 3,grImgY);

    SequentialFunction::addAbs(imgX, imgY, inputImage.getWidth(), inputImage.getHeight(), grImg);

    IntImage outputImage = IntImage(inputImage.getWidth(), inputImage.getHeight());


}