#pragma once

class GpuTimer {
private:
    cudaEvent_t start;
    cudaEvent_t stop;
public:
    GpuTimer();

    ~GpuTimer();

    void Start();

    void Stop();

    float Elapsed();
};
