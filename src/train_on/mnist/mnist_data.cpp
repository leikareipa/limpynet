/*
 * 2018 Tarpeeksi Hyvae Soft
 *
 * For loading and dealing with MNIST data.
 *
 */

#include <algorithm>
#include <string>
#include <vector>
#include <cstring>
#include "../../src/train_on/mnist/mnist_data.h"
#include "../../src/file/file.h"
#include "../../src/common.h"

mnist_data_c::mnist_data_c()
{
    /// FIXME: Filenames/path are hardcoded, for now.
    this->trainingImages = this->load_mnist_data("mnist/train-images.idx3-ubyte", 60000*28*28, 3);
    this->trainingLabels = this->load_mnist_data("mnist/train-labels.idx1-ubyte", 60000, 1);
    this->validationImages = this->load_mnist_data("mnist/t10k-images.idx3-ubyte", 10000*28*28, 3);
    this->validationLabels = this->load_mnist_data("mnist/t10k-labels.idx1-ubyte", 10000, 1);

    // The original images have values in the range 0..255. Convert them into the
    // range 0..1 for faster training.
    {
        const auto toReal = [](real &v){ v = (v / 255.0); };
        std::for_each(this->trainingImages.data.begin(), this->trainingImages.data.end(), toReal);
        std::for_each(this->validationImages.data.begin(), this->validationImages.data.end(), toReal);
    }

    return;
}

mnist_container_s mnist_data_c::load_mnist_data(const char *const filename, const uint numItems, const uint numDimensions)
{
    k_assert(((numDimensions == 1) || (numDimensions == 3)), "Only 1d and 3d IDX files are supported.");

    mnist_container_s mnistContents;

    const file_handle_t fh = kfile_open_file(filename, "rb");

    // Parse the header.
    {
        const uint magicNumber = kfile_read_value<u32>(fh, false);
        k_assert((( magicNumber & 0xffff0000)       == 0),    "Expected first two bytes of MNIST magic number to be 0.");
        k_assert((((magicNumber & 0x0000ff00) >> 8) == 0x08), "Expected the MNIST data to be of type unsigned byte.");

        /// TODO. Support data types other than u8. Also, test to make sure the
        /// file contains the same data type as what this template was called with.
        switch (numDimensions)
        {
            case 1:
            {
                mnistContents.elementCount = kfile_read_value<u32>(fh, false);
                k_assert((mnistContents.elementCount == numItems), "Unexpected data count in MNIST file.");

                break;
            }
            case 3:
            {
                mnistContents.elementCount = kfile_read_value<u32>(fh, false);
                mnistContents.rows = kfile_read_value<u32>(fh, false);
                mnistContents.cols = kfile_read_value<u32>(fh, false);
                k_assert(((mnistContents.elementCount * mnistContents.rows * mnistContents.cols) == numItems),
                         "Unexpected data count in MNIST file.");

                break;
            }
            default: k_assert(0, "Bad dimensions count."); break;
        }
    }

    // Load the data.
    {
        for (uint i = 0; i < mnistContents.elementCount; i++)
        {
            for (uint r = 0; r < (mnistContents.rows * mnistContents.cols); r++)
            {
                mnistContents.data.push_back(kfile_read_value<u8>(fh, false));
            }
        }
        k_assert((mnistContents.data.size() == numItems),
                 "Seem to have read an incorrect number of data points from the MNIST file.");
    }

    kfile_close_file(fh);

    return mnistContents;
}
