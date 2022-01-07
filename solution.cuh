#pragma once

#include "image.cuh"


class BaseSolution {
public:
    virtual PnmImage run(const PnmImage &inputImage, int argc, char **argv);
};