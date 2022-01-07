#pragma once

#include <cstdio>
#include <cstdint>
#include <vector_types.h>

class PnmImage {
private:
    uint32_t width, height;
    uchar3 *pixels;

    static float computeError(uchar3 *a1, uchar3 *a2, uint32_t n);

public:
    PnmImage() {
        width = 0;
        height = 0;
        pixels = nullptr;
    }

    PnmImage(uint32_t width, uint32_t height) {
        this->width = width;
        this->height = height;
        this->pixels = (uchar3 *) malloc(width * height * sizeof(uchar3));
    }

    PnmImage(const PnmImage &ref) {
        this->width = ref.width;
        this->height = ref.height;
        this->pixels = nullptr;
        if (ref.pixels) {
            pixels = (uchar3 *) malloc(width * height * sizeof(uchar3));
            memcpy(pixels, ref.pixels, width * height * sizeof(uchar3));
        }
    }

    PnmImage &operator=(const PnmImage &ref) {
        if (this == &ref)
            return *this;
        if (pixels)
            free(pixels);
        this->width = ref.width;
        this->height = ref.height;
        if (ref.pixels) {
            pixels = (uchar3 *) malloc(width * height * sizeof(uchar3));
            memcpy(pixels, ref.pixels, width * height * sizeof(uchar3));
        }
        return *this;
    }

    ~PnmImage() {
        if (pixels)
            free(pixels);
    }

    void read(const char *fileName);

    void write(const char *fileName);

    void compare(const PnmImage &other);

    uint32_t getWidth() const;


    uint32_t getHeight() const;


    uchar3 *getPixels() const;

};