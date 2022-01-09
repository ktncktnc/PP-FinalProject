#include "sequential_solution.cuh"
#include "utils.cuh"
#include <iostream>

using namespace std;

namespace SequentialFunction {
    void convolution(int *input, int inputWidth, int inputHeight, const int *filter, int filterSize, int* output) {
        int index, k_index, k_x, k_y, sum;

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
                        sum += input[k_index] * filter[(i + filterSize/2)*filterSize + j + filterSize/2];
                    }
                }
                output[index] = sum;
            }
        }
    }

    void convertToGray(uchar3* input, int width, int height, int* output){
        for(int i = 0; i < width * height; i++){
            output[i] = int(0.299f*input[i].x) + int(0.587f*input[i].y) + int(0.114f*input[i].z);
        }
    }

    void addAbs(int *input_1, int* input_2, int inputWidth, int inputHeight,
                 int *output) {
        int index, value;

        for(int x = 0; x < inputHeight; x++){
            for(int y = 0; y < inputWidth; y++) {
                index = x*inputWidth + y;
                value = abs(input_1[index]) + abs(input_2[index]);

                output[index] = value;
            }
        }
    }

    void createCumulativeEnergyMap(int * input, int inputWidth, int inputHeight, int* output) {
        int a, b, c;

        // Copy last line
        copyARow(input, inputWidth, inputHeight, 0, -1, output);

        for (int row = 1; row < inputHeight; row++) {
            for (int col = 0; col < inputWidth; col++) {
                a = output[(row - 1) * inputWidth + max(col - 1, 0)];
                b = output[(row - 1) * inputWidth + col];
                c = output[(row - 1) * inputWidth + min(col + 1, inputWidth - 1)];

                output[row * inputWidth + col] = input[row * inputWidth + col] + min(min(a, b), c);
            }
        }
    }
    void findSeamCurve(int* input, int inputWidth, int inputHeight, int* output){
        int a, b, c, min_idx, offset;
        min_idx = findMinIndex(input + (inputHeight - 1)*inputWidth, inputWidth);
        output[inputHeight - 1] = min_idx;

        for(int row = inputHeight - 2; row >= 0; row--){
            a = input[row*inputWidth + max(min_idx - 1, 0)];
            b = input[row*inputWidth + min_idx];
            c = input[row*inputWidth + min(min_idx + 1, inputWidth - 1)];

            if(min(a, b) > c) {
                offset = 1;
            }
            else if (min(b, c) > a){
                offset = -1;
            }
            else if (min(a, c) >= b)
                offset = 0;

            min_idx = min(max(min_idx + offset, 0), inputWidth - 1);
            output[row] = min_idx;
        }
    }

    // Util funcs--------------------
    int findMinIndex(int* arr, int size){
        int min_idx = 0;
        for(int i = 1; i < size; i++){
            if (arr[min_idx] > arr[i])
                min_idx = i;
        }

        return min_idx;
    }

    void copyARow(int* input, int width, int height, int rowIdx, int removedIdx, int* output){
        int output_idx = rowIdx * width, input_idx;

        for(int i = 0; i < width; i++){
            if(i == removedIdx) continue;

            input_idx = rowIdx * width + i;
            output[output_idx] = input[input_idx];
            output_idx++;
        }
    }

    void copyARow(uchar3* input, int width, int height, int rowIdx, int removedIdx, uchar3* output){
        int output_idx = rowIdx * width, input_idx;

        for(int i = 0; i < width; i++){
            if(i == removedIdx) continue;

            input_idx = rowIdx * width + i;
            output[output_idx] = input[input_idx];
            output_idx++;
        }
    }

    void reduce(uchar3* input, int width, int height, int* path, uchar3* output){
        for(int i = 0; i < height; i++){
            copyARow(input, width, height, i, path[i], output);
        }
    }
}

const int SequentialSolution::SOBEL_X[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
const int SequentialSolution::SOBEL_Y[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

PnmImage SequentialSolution::run(const PnmImage &inputImage, int argc, char **argv) {
    uchar3* input = inputImage.getPixels();
    uchar3* output;

    int cur_width = inputImage.getWidth();
    int height = inputImage.getHeight();

    printf("cur_width = %d\n", cur_width);

    for(int i = 0; i < 10; i++){
        SequentialSolution::scan(input, cur_width, height, output, i);

        cur_width--;

        input = output;


    }

    return PnmImage(cur_width, height, input);
}

void SequentialSolution::scan(uchar3* input, int width, int height, uchar3* output, int counter){
    int output_width = width - 1;
    int output_height = height;

    output = (uchar3*)malloc(output_width * output_height * sizeof(uchar3));

    // Convert to gray image
    int *grayImg = (int*)malloc(width * height * sizeof(int));
    SequentialFunction::convertToGray(input, width, height, grayImg);

    // Convolution
    int *gradX, *gradY, *grad;
    gradX = (int*) malloc(width * height * sizeof(int));
    gradY = (int*)malloc(width * height * sizeof(int));
    grad = (int*)malloc(width * height * sizeof(int));

    SequentialFunction::convolution(grayImg, width, height, SOBEL_X, FILTER_SIZE, gradX);
    SequentialFunction::convolution(grayImg, width, height, SOBEL_Y, FILTER_SIZE, gradY);

    // Cal energy
    SequentialFunction::addAbs(gradX, gradY, width, height, grad);

    if (counter == 0) {
        drawSobelImg(grad, width, height, "grad.pnm");
        drawSobelImg(gradX, width, height, "gradX.pnm");
        drawSobelImg(gradY, width, height, "gradY.pnm");
    }

    // Cal cumulative map
    int *map = (int*) malloc(width * height * sizeof(int));
    SequentialFunction::createCumulativeEnergyMap(grad, width, height, map);

    // Cal path
    int *path = (int*) malloc(height * sizeof(int));
    SequentialFunction::findSeamCurve(map, width, height, path);

    SequentialFunction::reduce(input, width, height, path, output);

    printf("width = %d height = %d ", width, height);

    if(counter == 0){
        writePnm(input, width, height, "input_img.pnm");

        writePnm(output, output_width, height, "debug_img.pnm");
    }

    free(grayImg);
    free(gradX);
    free(gradY);
    free(grad);
    free(map);
    free(path);
}

//IntImage SequentialSolution::scan(const PnmImage &inputImage) {
//    uchar* input = inputImage.getPixels();
//    uchar* output;
//
//    int cur_width = inputImage.getWidth();
//
//    for(int i = 0; i < 10; i++){
//        SequentialSolution::scan(input, cur_width, cur_height, output, i);
//
//        cur_width--;
//
//        input = output;
//    }
//
//    return new IntImage()
//}