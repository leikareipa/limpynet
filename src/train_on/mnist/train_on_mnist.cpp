/*
 * 2018 Tarpeeksi Hyvae Soft
 *
 */

#include <cstdio>
#include "../../src/train_on/mnist/mnist_data.h"
#include "../../src/train_on/train_on.h"
#include "../../src/nnetwork/nnetwork.h"
#include "../../src/cmd_line/cmd_line.h"

// Initialize the net for 28 x 28 images as input, and 10 (digits 0 through 9)
// for output. Also add any layers and parameters the user may have supplied on
// the command line.
bool k_initialize_net_for_user_data(nnetwork_c *const net, const int argc, char *const argv[])
{
    printf("Initializing for MNIST...\n");

    k_assert((net->num_layers() == 0), "Expected an empty net for initialization.");

    net->add_layer(784, activation_function_e::none);
    if (!k_parse_command_line(argc, argv, net))
    {
        return false;
    }
    net->add_layer(10, activation_function_e::softmax);

    return true;
}

// Display random digits from the MNIST validation set in the terminal, and for
// each image show the user the net's prediction for the label.
static void quiz(nnetwork_c &net, const mnist_data_c &mnistSet)
{
    printf("<Press enter to start the quiz, or CTRL+C to quit.>\n");
    getchar();

    while (1)
    {
        const auto &imageSource = mnistSet.validationImages;
        const auto &labelSource = mnistSet.validationLabels;

        const uint imageIdx = (net.random_number() * imageSource.num_elements());
        const auto image = imageSource.contents_of_element(imageIdx);
        const auto label = labelSource.contents_of_element(imageIdx).at(0);

        net.propagate(image);
        const uint predictedLabel = net.strongest_output_neuron_idx();

        // Draw the digit into the terminal.
        for (uint y = 3; y < (imageSource.rows - 1); y++)
        {
            for (uint x = 0; x < imageSource.cols; x++)
            {
                const int ch = image.at(x + y * imageSource.cols) * 255;
                printf("%c", (ch < 30)? ' ' : (ch < 150)? '.' : (ch < 220)? '*' : '#');
            }
            printf("\n");
        }

        printf("The net guesses %d. That's %s.", predictedLabel, ((predictedLabel == label)? "correct" : "wrong"));
        getchar();
    }

    return;
}

bool k_train_net_on_user_data(nnetwork_c *const net)
{
    mnist_data_c mnistSet;

    printf("Training on MNIST (%d/%d)...\n",
           mnistSet.trainingImages.num_elements(), mnistSet.validationImages.num_elements());

    for (uint i = 0; i < net->num_training_epochs(); i++)
    {
        // Test the net on MNIST images that it won't see during training.
        uint numValidationCorrect = 0;
        {
            const auto &imageSource = mnistSet.validationImages;
            const auto &labelSource = mnistSet.validationLabels;

            for (uint m = 0; m < imageSource.num_elements(); m++)
            {
                const uint imageIdx = (net->random_number() * imageSource.num_elements());
                const auto image = imageSource.contents_of_element(imageIdx);

                // The expected output is a vector where all values are zero except
                // for that of the nth element, where n = the image's category number.
                std::vector<real> expectedOutput;
                expectedOutput.resize(mnistSet.numCategories, 0);
                expectedOutput.at(int(labelSource.contents_of_element(imageIdx).at(0))) = 1;

                // Pass the image through the net, and compare its output to what was expected.
                net->propagate(image);
                if (net->activation_vector() == expectedOutput)
                {
                    numValidationCorrect++;
                }
            }
        }

        // Train the net.
        /// FIXME. Needlessly duplicates code from above.
        uint numTrainingCorrect = 0;
        {
            const auto &imageSource = mnistSet.trainingImages;
            const auto &labelSource = mnistSet.trainingLabels;

            // Loop through about each of the MNIST training images.
            for (uint m = 0; m < imageSource.num_elements(); m++)
            {
                const uint imageIdx = (net->random_number() * imageSource.num_elements());
                const auto image = imageSource.contents_of_element(imageIdx);

                std::vector<real> expectedOutput;
                expectedOutput.resize(mnistSet.numCategories, 0);
                expectedOutput.at(int(labelSource.contents_of_element(imageIdx).at(0))) = 1;

                // See whether the net as-is can correctly identify this image.
                net->propagate(image);
                if (net->activation_vector() == expectedOutput)
                {
                    numTrainingCorrect++;
                }

                // Then train it some more on it.
                net->train(image, expectedOutput);
            }
        }

        const real trainingAccuracy = ((numTrainingCorrect / (real)mnistSet.trainingImages.num_elements()) * 100);
        const real validationAccuracy = ((numValidationCorrect / (real)mnistSet.validationImages.num_elements()) * 100);

        printf("Epoch %d of %d: train = %.3f%%, validate = %.3f%%.\n",
               (i + 1), net->num_training_epochs(), trainingAccuracy, validationAccuracy);
    }

    printf("Training finished.\n");

    quiz(*net, mnistSet);

    return true;
}
