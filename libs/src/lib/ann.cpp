#include "ann.hpp"

#include <algorithm>
#include <iostream>

using namespace std;

NeuralNet::NeuralNet(vector<unsigned int> numberOfNeuronsPerLayer) :topology(move(numberOfNeuronsPerLayer)) {

    int numberOfLayers = this->topology.size();
    this->weights.reserve(numberOfLayers - 1); //the first (input) layer hasn't weights

    for(int layerIndex = 1; layerIndex < numberOfLayers; layerIndex ++) {

        int currentLayerSize = this->topology[layerIndex];
        int previousLayerSize = this->topology[layerIndex - 1];

        vector<vector<double>> layer(currentLayerSize);

        auto gen = [previousLayerSize]() {
            vector<double> result(previousLayerSize + 1); // +1 for the bias weight
            generate(result.begin(), result.end(), []() {
                return (static_cast<double>(rand()) / RAND_MAX) * .1 - .05; //provides a unifor disto random value into [-0.05, +0.05[
            });
            return result;
        };

        generate(layer.begin(), layer.end(), gen);

        this->weights.push_back(layer);

    }

}

NeuralNet::NeuralNet(vector<vector<vector<double>>> initialWeights) {

    this->weights = move(initialWeights);
    int numberOfLayers  = this->weights.size() + 1; // the input layer hasn't weights
    for(int layerIndex = 0; layerIndex < numberOfLayers - 1; layerIndex++) {
        int numberOfNeurons = this->weights[layerIndex][0].size() - 1; //-1 because the weight vector has +1 number of weights for the bias
        this->topology.push_back(numberOfNeurons);
    }

    int lastLayersize = this->weights[this->weights.size() - 1].size();
    this->topology.push_back(lastLayersize);
}

double sigmoid(double value) {
    double result;
    if(value > 45) {
        result = 1;
    } else if(value < -45) {
        result = 0;
    } else {
        result = 1.0 / (1 + exp(-value));
    }
    return result;
}

vector<double> NeuralNet::forward(vector<double> & currentInput, int layerIndex) {
    vector<vector<double>> & layer = this->weights[layerIndex - 1]; // -1 because the input layer hasn' t weights
    currentInput.push_back(1); //1 for the bias value
    int numberOfNeurons = this->topology[layerIndex];
    vector<double> output(numberOfNeurons);

    auto neuronValue = [&currentInput] (vector<double> & neuronWeights) {
        double value = inner_product(neuronWeights.begin(), neuronWeights.end(), currentInput.begin(), 0.0);
        double result = sigmoid(value);
        return result;
    };

    transform(layer.begin(), layer.end(), output.begin(), neuronValue);
    return output;
}

vector<double> NeuralNet::forward(const std::vector<double> & input) {

    vector<double> currentInput(input);

    int numberOfLayers = this->topology.size();

    for(int layerIndex = 1; layerIndex < numberOfLayers; layerIndex ++) {
        currentInput = this->forward(currentInput, layerIndex);
    }

    return currentInput;
}

void NeuralNet::train(vector<vector<double>> & dataset, int outputIndex, double learningRate, int epochs) {

    const int datasetSize = dataset.size();
    const int numberOfLayers = this->topology.size();

    for(int epoch = 0; epoch < epochs; epoch++) {

        //random_shuffle(dataset.begin(), dataset.end());

        for(int n = 0; n < datasetSize; n++) {
            //{.1, .5, .2, .8, 1.0, 0.0}
            vector<double> & instance = dataset[n];
            vector<double> currentInput(instance.begin(), instance.begin() + outputIndex);
            vector<double> expectedOutput(instance.begin() + outputIndex, instance.end());

            //forward
            vector<vector<double>> inputPerLayer;
            for(int layerIndex = 1; layerIndex < numberOfLayers; layerIndex ++) {
                inputPerLayer.push_back(currentInput);
                currentInput = this->forward(currentInput, layerIndex);
            }
            vector<double> output(currentInput);
            //backward

            vector<double> error(expectedOutput);
            transform(error.begin(), error.end(), output.begin(), error.begin(), std::minus<double>());
            vector<double> sigma(error);

            transform(sigma.begin(), sigma.end(), output.begin(), sigma.begin(), [](double err, double actual){
                double delta = actual * (1 - actual);
                return err * delta;
            });

            for(int layerIndex = (numberOfLayers - 1); layerIndex > 0; layerIndex--) {

                vector<vector<double>> & layer = this->weights[layerIndex - 1];
                vector<double> & input = inputPerLayer[layerIndex - 1];
                input.push_back(1); //the bias input value

                int numberOfNeuronsInPreviousLayer = this->topology[layerIndex - 1];
                int numberOfneurons = this->topology[layerIndex];

                vector<double> newSigma;
                for(int j = 0; j < numberOfNeuronsInPreviousLayer; j++) {
                    double newSigmaValue = 0.0;

                    for(int s = 0, size = sigma.size(); s < size; s++) {
                        newSigmaValue = newSigmaValue + sigma[s] * layer[s][j];
                    }
                    newSigmaValue = newSigmaValue * input[j] * (1 - input[j]);
                    newSigma.push_back(newSigmaValue);
                }

                //weight update

                for(int j = 0; j < numberOfneurons; j++) {

                    vector<double> & neuronWeight = layer[j];
                    double factor = learningRate * sigma[j];
                    transform(neuronWeight.begin(), neuronWeight.end(), input.begin(), neuronWeight.begin(), [factor](double oldWeight, double inputValue) {
                        return oldWeight + (factor * inputValue);
                    });

                }

                sigma = move(newSigma);
            }

        }

    }

}