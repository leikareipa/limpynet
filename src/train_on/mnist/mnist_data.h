/*
 * 2018 Tarpeeksi Hyvae Soft
 *
 * For loading and dealing with MNIST data.
 *
 */

#ifndef MNIST_DATA_H
#define MNIST_DATA_H

#include <vector>
#include "../../src/types.h"

// Contains raw data loaded from a MNIST file; and provides ordered access to it.
struct mnist_container_s
{
    // All of the contents as a flat array.
    std::vector<real> data;

    // The number of discrete elements; in this case, MNIST images/labels.
    uint elementCount = 0;

    // X,y dimensions of each element. For labels, this is ignored, but
    // for images, this is its width and height.
    uint rows = 1;
    uint cols = 1;

    void operator=(const mnist_container_s &other)
    {
        this->elementCount = other.elementCount;
        this->rows = other.rows;
        this->cols = other.cols;
        this->data = other.data;
    }

    uint num_elements(void) const
    {
        return this->elementCount;
    }

    // Returns a copy of the idx'th element's data.
    std::vector<real> contents_of_element(const uint idx) const
    {
        std::vector<real> c;

        const uint offset = (idx * this->rows * this->cols);

        for (uint y = 0; y < this->rows; y++)
        {
            for (uint x = 0; x < this->cols; x++)
            {
                c.push_back(this->data.at(offset + (x + y * this->cols)));
            }
        }

        return c;
    }
};

// Pools together the training and validation image/label sets in MNIST.
class mnist_data_c
{
public:
    mnist_data_c();

    // Ten categories, for the digits 0 through 9.
    const int numCategories = 10;

    mnist_container_s trainingImages;
    mnist_container_s trainingLabels;
    mnist_container_s validationImages;
    mnist_container_s validationLabels;

private:
    // Loads data in the MNIST IDX format from the given file. Will expect the given
    // number of items to be found, and for the file to have the given dimensionality.
    /// Note that at the moment, only works with u8 data, and will assume that files
    /// with one dimension contain image labels, and files with three dimensions have
    /// the images themselves. In other words, this isn't a general IDX format reader.
    mnist_container_s load_mnist_data(const char *const filename, const uint numItems, const uint numDimensions);
};

#endif
