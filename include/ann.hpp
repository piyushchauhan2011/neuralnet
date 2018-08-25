#ifndef ANN_H
#define ANN_H

#include <vector>
#include <ostream>

class NeuralNet {

    private:
        std::vector<unsigned int> topology;
        std::vector<std::vector<std::vector<double>>> weights;
        std::vector<double> forward(std::vector<double> & input, int layerIndex);

    public:
        NeuralNet(std::vector<unsigned int> numberOfNeuronsPerLayer);
        NeuralNet(std::vector<std::vector<std::vector<double>>> initialWeights);
        virtual ~NeuralNet() {};

        std::vector<double> forward(const std::vector<double> & input);

        /**
         * The netowork is trainned by the backpropagation algorithm with sequential mode
         * 
         * */

        void train(std::vector<std::vector<double>> & dataset, int outputIndex, double learningRate, int epochs);

};


#endif