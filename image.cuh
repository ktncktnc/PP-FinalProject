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

    PnmImage(uint32_t width, uint32_t height, uchar3* pixels) {
        this->width = width;
        this->height = height;
        this->pixels = pixels;
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

class IntImage {
private:
    uint32_t width, height;
    int32_t *pixels;

public:
    uint32_t getWidth() const;

    uint32_t getHeight() const;

    int32_t *getPixels() const;

    IntImage() {
        width = 0;
        height = 0;
        pixels = nullptr;
    }

    IntImage(uint32_t width, uint32_t height) {
        this->width = width;
        this->height = height;
        this->pixels = (int32_t *) malloc(width * height * sizeof(int32_t));
    }

    IntImage(const IntImage &ref) {
        this->width = ref.width;
        this->height = ref.height;
        this->pixels = nullptr;
        if (ref.pixels) {
            pixels = (int32_t *) malloc(width * height * sizeof(int32_t));
            memcpy(pixels, ref.pixels, width * height * sizeof(int32_t));
        }
    }

    IntImage &operator=(const IntImage &ref) {
        if (this == &ref)
            return *this;
        if (pixels)
            free(pixels);
        this->width = ref.width;
        this->height = ref.height;
        if (ref.pixels) {
            pixels = (int32_t *) malloc(width * height * sizeof(int32_t));
            memcpy(pixels, ref.pixels, width * height * sizeof(int32_t));
        }
        return *this;
    }

    explicit operator PnmImage() const {
        PnmImage outputImage = PnmImage(this->width, this->height);
        if (!this->pixels) return outputImage;
        for (int i = 0; i < this->height * this->width; ++i) {
            int32_t temp = this->pixels[i];
            if (temp > 255)
                temp = 255;
            if (temp < 0)
                temp = 0;
            outputImage.getPixels()[i] = make_uchar3((char) temp, (char) temp,
                                                     (char) temp);
        }
        return outputImage;
    }


    ~IntImage() {
        if (pixels)
            free(pixels);
    }
};