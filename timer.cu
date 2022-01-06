#include "timer.cuh"

GpuTimer::GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

GpuTimer::~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void GpuTimer::Start() {
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);
}

void GpuTimer::Stop() {
    cudaEventRecord(stop, 0);
}

float GpuTimer::Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
}