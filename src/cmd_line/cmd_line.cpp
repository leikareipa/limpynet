/*
 * 2018 Tarpeeksi Hyvae Soft
 *
 */

#include <cstdlib>
#include <unistd.h>
#include "../../src/nnetwork/nnetwork.h"

bool k_parse_command_line(const int argc, char *const argv[], nnetwork_c *const net)
{
    int c = 0;
    while ((c = getopt(argc, argv, "R:L:T:G:N:S:e:r:x")) != -1)
    {
        switch (c)
        {
            case 'x':
            {
                printf("Running XOR test... "); fflush(stdout);
                printf("XOR test result: %.3f%%\n", nnetwork_c::xor_test());

                break;
            }
            case 'S':
            {
                const uint numNeurons = strtol(optarg, NULL, 10);
                net->add_layer(numNeurons, activation_function_e::softmax);

                break;
            }
            case 'N':
            {
                const uint numNeurons = strtol(optarg, NULL, 10);
                net->add_layer(numNeurons, activation_function_e::none);

                break;
            }
            case 'R':
            {
                const uint numNeurons = strtol(optarg, NULL, 10);
                net->add_layer(numNeurons, activation_function_e::relu);

                break;
            }
            case 'L':
            {
                const uint numNeurons = strtol(optarg, NULL, 10);
                net->add_layer(numNeurons, activation_function_e::leaky_relu);

                break;
            }
            case 'T':
            {
                const uint numNeurons = strtol(optarg, NULL, 10);
                net->add_layer(numNeurons, activation_function_e::tanh_sigmoid);

                break;
            }
            case 'G':
            {
                const uint numNeurons = strtol(optarg, NULL, 10);
                net->add_layer(numNeurons, activation_function_e::log_sigmoid);

                break;
            }
            case 'e':
            {
                const uint numTrainingEpochs = strtol(optarg, NULL, 10);
                net->set_num_training_epochs(numTrainingEpochs);

                break;
            }
            case 'r':
            {
                real learningRate = strtod(optarg, NULL);
                if (learningRate <= 0)
                {
                    NBENE(("Invalid learning rate: %f.", learningRate));
                    learningRate = 0.001;
                }

                net->set_learning_rate(learningRate);

                break;
            }
        }
    }

    return true;
}

