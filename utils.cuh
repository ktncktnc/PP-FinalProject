#pragma once
#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

void printDeviceInfo();

void writePnm(uchar3* pixels, int width, int height, char *fileName);

void drawSobelImg(int* dImg, int width, int height, char* savePath);
