/*
 * 2016, 2018 Tarpeeksi Hyvae Soft
 *
 * A simple backpropagating feedforward neural network.
 *
 */

#include <functional>
#include <algorithm>
#include "../../src/nnetwork/nnetwork.h"
#include "../../src/common.h"

nnetwork_c::nnetwork_c()
{
    unsigned randSeed = std::chrono::system_clock::now().time_since_epoch().count();
    this->randomNumberGenerator.seed(randSeed);

    return;
}

nnetwork_c::~nnetwork_c()
{
    delete this->randomNormalDistribution;
    delete this->randomUniformDistribution;

    return;
}

void nnetwork_c::add_layer(const uint numNeurons, const activation_function_e functionType)
{
    neuron_layer_s newLayer;

    uint precedingLayerSize = 0;
    if (!this->layers.empty())
    {
        precedingLayerSize = this->layers.back().neurons.size();
    }

    // Add the desired number of neurons, and initialize them to their default values, including with randomized weights.
    newLayer.neurons.resize(numNeurons, neuron_s(precedingLayerSize, randomNumberGenerator, randomNormalDistribution));
    newLayer.activationFunction = functionType;

    this->layers.push_back(newLayer);

    return;
}

void nnetwork_c::set_inputs(const std::vector<real> inputs)
{
    if (this->layers.empty() ||
        (inputs.size() != this->layers.front().neurons.size()))
    {
        NBENE(("Incompatible input layer for the given inputs."));
        return;
    }

    for (size_t i = 0; i < this->layers.front().neurons.size(); i++)
    {
        this->layers.front().neurons.at(i).output = inputs.at(i);
    }

    return;
}

void nnetwork_c::set_expected_output(const std::vector<real> expected)
{
    if (expected.size() != this->layers.back().neurons.size())
    {
        NBENE(("Number of expected output elements does not match the number of output neurons."));
        return;
    }

    this->expectedOutput = expected;

    return;
}

void nnetwork_c::apply_softmax_to_output()
{
    // Find the highest output value among the output neurons, for stabilizing potential numerical issues with softmax.
    real maxOutput = 0;
    {
        std::vector<real> thing;

        for (size_t i = 0; i < this->layers.back().neurons.size(); i++)
        {
            thing.push_back(this->output_of_neuron(i));
        }

        auto min = std::minmax_element(thing.begin(), thing.end());

        maxOutput = *(min.second);
    }

    // Calculate exponents for each output neuron.
    real expSum = 0;
    std::vector<real> expOutputs;
    for (size_t i = 0; i < this->layers.back().neurons.size(); i++)
    {
        expOutputs.push_back(exp(this->output_of_neuron(i) - maxOutput));
        expSum += expOutputs.back();
    }

    // Apply the softmax function to all output neurons.
    for (size_t i = 0; i < this->layers.back().neurons.size(); i++)
    {
        this->layers.back().neurons.at(i).output = (expOutputs.at(i) / expSum);
    }

    return;
}

bool nnetwork_c::announce_current_configuration(void) const
{
    printf("Net:");

    // Topology.
    {
        printf("\tTopology: ");

        for (const auto layer: this->layers)
        {
            switch (layer.activationFunction)
            {
            case activation_function_e::leaky_relu:   printf("L"); break;
            case activation_function_e::relu:         printf("R"); break;
            case activation_function_e::log_sigmoid:  printf("G"); break;
            case activation_function_e::tanh_sigmoid: printf("T"); break;
            case activation_function_e::softmax:      printf("S"); break;
            case activation_function_e::none:         printf("N"); break;
            default: printf("?"); break;
            }

            printf("%d-", (int)layer.neurons.size());
        }

        printf("\b \n");
    }

    // Miscellaneous info.
    {
        printf("\tLearning rate: %f\n", this->learningRate);
        printf("\tTraining epochs: %d\n", this->numTrainingEpochs);
    }

    return true;
}

uint nnetwork_c::num_training_epochs() const
{
    return this->numTrainingEpochs;
}

uint nnetwork_c::num_layers() const
{
    return this->layers.size();
}

uint nnetwork_c::strongest_output_neuron_idx(void)
{
    // Find the node with the strongest activation.
    int strongestNeuronIdx = -1;
    real strongestActivation = -1;
    for (size_t i = 0; i < this->layers.back().neurons.size(); i++)
    {
        if (this->output_of_neuron(i) > strongestActivation)
        {
            strongestActivation = this->output_of_neuron(i);
            strongestNeuronIdx = i;
        }
    }

    return strongestNeuronIdx;
}

real nnetwork_c::xor_test(void)
{
    nnetwork_c xorNet;
    xorNet.add_layer(2, activation_function_e::none);
    xorNet.add_layer(4, activation_function_e::tanh_sigmoid);
    xorNet.add_layer(1, activation_function_e::tanh_sigmoid);
    xorNet.set_learning_rate(0.01);
    xorNet.set_num_training_epochs(50000);

    std::vector<real> input;
    input.resize(2, 0);

    std::vector<real> expectedOutput;
    expectedOutput.resize(1, 0);

    uint loopsCorrect = 0;
    const uint numLoops = 200;
    for (uint loops = 0; loops < numLoops; loops++)
    {
        // Train the net on XOR.
        for (uint i = 0; i < xorNet.num_training_epochs(); i++)
        {
            const std::vector<std::function<void()>> conditions =
                    {[&]{ input.at(0) = 1; input.at(1) = 1; expectedOutput.at(0) = 0; },
                     [&]{ input.at(0) = 1; input.at(1) = 0; expectedOutput.at(0) = 1; },
                     [&]{ input.at(0) = 0; input.at(1) = 0; expectedOutput.at(0) = 0; },
                     [&]{ input.at(0) = 0; input.at(1) = 1; expectedOutput.at(0) = 1; }};

            for (auto set_conditions: conditions)
            {
                set_conditions();
                xorNet.train(input, expectedOutput);
            }
        }

        // Test to see whether the net learnt XOR.
        int numCorrect = 0;
        {
            input.at(0) = 0;
            input.at(1) = 0;
            xorNet.propagate(input);
            numCorrect += !xorNet.output_neuron_fires(0);

            input.at(0) = 1;
            input.at(1) = 1;
            xorNet.propagate(input);
            numCorrect += !xorNet.output_neuron_fires(0);

            input.at(0) = 1;
            input.at(1) = 0;
            xorNet.propagate(input);
            numCorrect += xorNet.output_neuron_fires(0);

            input.at(0) = 0;
            input.at(1) = 1;
            xorNet.propagate(input);
            numCorrect += xorNet.output_neuron_fires(0);
        }

        loopsCorrect += bool(numCorrect == 4);
    }

    return ((loopsCorrect / (real)numLoops) * 100);
}

void nnetwork_c::propagate_forward()
{
    // Loop for each layer (ignoring the input layer).
    for (size_t i = 1; i < this->layers.size(); i++)
    {
        // Loop for each neuron in the layer.
        for (size_t o = 0; o < this->layers.at(i).neurons.size(); o++)
        {
            real inputSum = this->layers.at(i).neurons.at(o).biasWeight;

            // Loop for each weight in the neuron, summing up the inputs from the preceding layer. Note that q here
            // corresponds both to the weight index of the current neuron and the index of the neuron in the preceding
            // layer, since the number of weights is equal to the number of neurons in the preceding layer.
            for (size_t q = 0; q < layers.at(i).neurons.at(o).inputWeights.size(); q++)
            {
                inputSum += (this->layers.at(i-1).neurons.at(q).output * this->layers.at(i).neurons.at(o).inputWeights.at(q));
            }

            // The output of this neuron is decided by passing its sum of inputs through an activation function.
            this->layers.at(i).neurons.at(o).output = this->activation_function(inputSum, this->layers.at(i).activationFunction);
        }
    }

    // For the softmax activation function on the output layer, we need to collect the output of all output neurons before applying
    // the softmax function. That's why we do it here, after the forward propagation step has finished.
    if (this->layers.back().activationFunction == activation_function_e::softmax)
    {
        this->apply_softmax_to_output();
    }

    return;
}

void nnetwork_c::propagate_back()
{
    // Calculate the error terms at the output neurons.
    {
        auto &outputLayer = this->layers.back();

        for (size_t i = 0; i < outputLayer.neurons.size(); i++)
        {
            outputLayer.neurons.at(i).delta =
                    (this->activation_function_derivative(outputLayer.neurons.at(i).output, outputLayer.activationFunction) *
                                                          (outputLayer.neurons.at(i).output - this->expectedOutput.at(i)));
        }
    }

    // Backpropagate the error terms from the output layer to the preceding layers. We ignore the first (input) layer, since we
    // don't need to compute its error terms. We also ignore the last (output) layer, since its error term was calculated above.
    for (size_t i = (this->layers.size() - 2); i >= 1; i--)
    {
        auto &thisLayer = this->layers.at(i);
        auto &nextLayer = this->layers.at(i+1);

        // Loop through all neurons in this layer.
        for (size_t o = 0; o < thisLayer.neurons.size(); o++)
        {
            real deltaSum = 0;

            // Loop through all neurons in the following layer, summing up their error deltas weighted by their connection to this
            // neuron. Note that the oth input weight of the neuron in the following layer corresponds to the oth neuron in the
            // current layer.
            for (size_t q = 0; q < nextLayer.neurons.size(); q++)
            {
                deltaSum += nextLayer.neurons.at(q).delta * nextLayer.neurons.at(q).inputWeights.at(o);
            }

            // Assign the error function to this neuron.
            thisLayer.neurons.at(o).delta = (activation_function_derivative(thisLayer.neurons.at(o).output, thisLayer.activationFunction) * deltaSum);
        }
    }

    return;
}

real nnetwork_c::loss_function()
{
    if (layers.empty())
    {
        NBENE(("Attempted to fetch loss function for an empty network; not allowing this."));
        return -1;
    }

    real loss = 0;
    for (size_t i = 0; i < layers.back().neurons.size(); i++)
    {
        loss += powf(this->output_of_neuron(i) - expectedOutput.at(i),2);
    }

    return (loss / layers.back().neurons.size());
}

std::vector<std::vector<real>> nnetwork_c::get_weights_in_layer(const uint layer)
{
    if (layer >= this->layers.size())
    {
        NBENE(("Attempted to access out of bounds; not allowing it."));
        return {};
    }

    std::vector<std::vector<real>> weights;
    for (const auto neuron: this->layers.at(layer).neurons)
    {
        weights.push_back(neuron.inputWeights);
    }

    return weights;
}

void nnetwork_c::update_weights()
{
    // Loop through each layer, updating their weights based on the deltas calculated in the backpropagation
    // step. Note that we skip the input layer, as it has no incoming connections.
    for (size_t i = 1; i < this->layers.size(); i++)
    {
        auto &thisLayer = this->layers.at(i);
        auto &prevLayer = this->layers.at(i-1);

        for (size_t o = 0; o < thisLayer.neurons.size(); o++)
        {
            for (size_t p = 0; p < thisLayer.neurons.at(o).inputWeights.size(); p++)
            {
                const real gradient = (prevLayer.neurons.at(p).output * thisLayer.neurons.at(o).delta);

                thisLayer.neurons.at(o).inputWeights.at(p) += -learningRate * gradient;
            }

            thisLayer.neurons.at(o).biasWeight += (-learningRate * thisLayer.neurons.at(o).delta);
        }
    }

    return;
}

real nnetwork_c::train(const std::vector<real> input, const std::vector<real> expectedOutput)
{
    this->set_inputs(input);
    this->set_expected_output(expectedOutput);

    this->propagate_forward();
    this->propagate_back();
    this->update_weights();

    return this->loss_function();
}

std::vector<real> nnetwork_c::activation_vector(void)
{
    if (this->layers.empty())
    {
        NBENE(("Cannot return an activation matrix for an empty network."));
        return {};
    }

    std::vector<real> activations;
    activations.resize(this->layers.back().neurons.size(), 0);

    // Find the node with the strongest activation.
    int strongestNeuron = 0;
    real strongestActivation = -1;
    for (size_t i = 0; i < this->layers.back().neurons.size(); i++)
    {
        if (this->output_of_neuron(i) > strongestActivation)
        {
            strongestActivation = this->output_of_neuron(i);
            strongestNeuron = i;
        }
    }

    // If the strongest neuron produces an output above the activation threshold,
    // consider that to be the active neuron.
    if (this->output_neuron_fires(strongestNeuron))
    {
        activations.at(strongestNeuron) = 1;
    }

    return activations;
}

void nnetwork_c::propagate(const std::vector<real> input)
{
    this->set_inputs(input);
    this->propagate_forward();

    return;
}

real nnetwork_c::random_number(void)
{
    return this->randomUniformDistribution->operator()(this->randomNumberGenerator);
}

real nnetwork_c::activation_function(const real sum, const activation_function_e functionType) const
{
    switch (functionType)
    {
        case activation_function_e::log_sigmoid:   return this->af_logsigmoid(sum);
        case activation_function_e::relu:          return this->af_relu(sum);
        case activation_function_e::leaky_relu:    return this->af_leakyrelu(sum);
        case activation_function_e::tanh_sigmoid:  return this->af_tanhsigmoid(sum);
        case activation_function_e::mtanh_sigmoid: return this->af_modtanhsigmoid(sum);
        case activation_function_e::softmax:       return sum; /// Temp hack. We'll calculate the softmax elsewhere, for now, so just return here.
        default: NBENE(("Failed to find an activation function for id %d.", (int)functionType)); return -1;
    }
}

real nnetwork_c::activation_function_derivative(const real output, const activation_function_e functionType) const
{
    switch (functionType)
    {
        case activation_function_e::log_sigmoid:   return this->af_logsigmoid_deriv(output);
        case activation_function_e::relu:          return this->af_relu_deriv(output);
        case activation_function_e::leaky_relu:    return this->af_leakyrelu_deriv(output);
        case activation_function_e::tanh_sigmoid:  return this->af_tanhsigmoid_deriv(output);
        case activation_function_e::mtanh_sigmoid: return this->af_modtanhsigmoid_deriv(output);
        case activation_function_e::softmax:       return 1; /// FIXME. Assumes, without knowing, that the softmax will be on the output layer only.
        default: NBENE(("Failed to find an activation function for id %d.", (int)functionType)); return -1;
    }
}
