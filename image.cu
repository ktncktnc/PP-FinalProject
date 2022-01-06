#include "image.cuh"

void PnmImage::read(const char *fileName) {
    FILE *f = fopen(fileName, "r");
    if (f == nullptr) {
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    char type[3];
    fscanf(f, "%s", type);

    if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
    {
        fclose(f);
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fscanf(f, "%i", &width);
    fscanf(f, "%i", &height);

    int max_val;
    fscanf(f, "%i", &max_val);
    if (max_val > 255) // In this exercise, we assume 1 byte per value
    {
        fclose(f);
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    pixels = (uchar3 *) malloc(width * height * sizeof(uchar3));
    for (int i = 0; i < width * height; i++)
        fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

    fclose(f);
}

void PnmImage::write(const char *fileName) {
    FILE *f = fopen(fileName, "w");
    if (f == nullptr) {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fprintf(f, "P3\n%i\n%i\n255\n", width, height);

    for (int i = 0; i < width * height; i++)
        fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);

    fclose(f);
}

void PnmImage::compare(const PnmImage &other) {
    float err = 0;
    if (this->height != other.height || this->width != other.width)
        err = float(this->width * this->height * 255);
    else
        err = computeError(this->pixels, other.pixels, width * height);
    printf("Error: %f\n", err);
}

float PnmImage::computeError(uchar3 *a1, uchar3 *a2, uint32_t n) {
    long long err = 0;
    for (int i = 0; i < n; i++) {
        err += abs((int) a1[i].x - (int) a2[i].x);
        err += abs((int) a1[i].y - (int) a2[i].y);
        err += abs((int) a1[i].z - (int) a2[i].z);
    }
    return float(err) / float(n * 3);
}

uint32_t PnmImage::getWidth() const {
    return width;
}

uint32_t PnmImage::getHeight() const {
    return height;
}


uchar3 *PnmImage::getPixels() const {
    return pixels;
}



