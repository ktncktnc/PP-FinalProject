#include "utils.cuh"
#include <cstdio>

void printDeviceInfo() {
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);
    printf("****************************\n");
}

void writePnm(uchar3* pixels, int width, int height, char *fileName){
    FILE * f = fopen(fileName, "w");
    if (f == NULL)
    {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fprintf(f, "P3\n%i\n%i\n255\n", width, height);

    for (int i = 0; i < width * height; i++)
        fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);

    fclose(f);
}

void drawSobelImg(int* dImg, int width, int height, char* savePath){
    uchar3* img2draw = (uchar3 *)malloc(width * height * sizeof(uchar3));
    for(int y = 0; y < height; y++){
        for(int x = 0; x< width; x++){
            int idx = y*width+x ;
            img2draw[idx] = make_uchar3(min(int(dImg[idx]), 255), min(int(dImg[idx]), 255), min(int(dImg[idx]), 255));
        }
    }
    writePnm(img2draw, width, height, savePath);
}