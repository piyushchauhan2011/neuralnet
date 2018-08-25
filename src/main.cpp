#include "ann.hpp"
#include "dataio.hpp"

#include <iostream>
#include <algorithm>

using namespace std;

void evaluate(NeuralNet & net, vector<vector<double>> & dataset, size_t numberOfInputs);

int main(int argc, char ** argv) {

    srand(time(NULL));

    cout << "Getting started!\n"; 

    NeuralNet net({4, 3, 2});

    vector<vector<double>> dataset;

    size_t numberOfInputs, numberOfOutputs;

    bool read = loadData("../data/sampling.txt", dataset, numberOfInputs, numberOfOutputs);

    if(read) {
        int datasetSize = dataset.size();
        int trainingDatasetSize = datasetSize * 66 / 100;

        //random_shuffle(dataset.begin(), dataset.end());

        vector<vector<double>> trainingDataset(dataset.begin(), dataset.begin() + trainingDatasetSize);
        vector<vector<double>> evaluationDataset(dataset.begin() + trainingDatasetSize, dataset.end());

        net.train(trainingDataset, numberOfInputs, 0.1, 500);

        evaluate(net, evaluationDataset, numberOfInputs);

    } else {
        cout << "failed to load the data";
    }
    cout << "finish\n";

    return 0;

}

void evaluate(NeuralNet & net, vector<vector<double>> & dataset, size_t numberOfInputs) {

    int datasetSize = dataset.size();

    int TP = 0;
    int TN = 0;
    int FP = 0;
    int FN = 0;

    double accErr = 0.0;

    for(int n = 0; n < datasetSize; n++) {

        vector<double> & instance = dataset[n];
        vector<double> currentInput(instance.begin(), instance.begin() + numberOfInputs);
        vector<double> expectedOutput(instance.begin() + numberOfInputs, instance.end());

        vector<double> output = net.forward(currentInput);

        double expectedClass = expectedOutput[1];
        double score = output[1];

        if(expectedClass > .5) {

            if(score > .5) {
                TP = TP + 1;
            } else {
                FN = FN + 1;
            }

        } else {

            if(score > .5) {
                FP = FP + 1;
            } else {
                TN = TN + 1;
            }
        }

        double sumErr = pow(expectedOutput[0] - output[0], 2) + pow(expectedOutput[1] - output[1], 2);
        accErr = accErr + sumErr / 2.0;

    }

    double rmse = sqrt(accErr / datasetSize);

    double precision = (double) TP / (TP + FP);
    double recall = (double) TP / (TP + FN);

    double TN_rate = (double) TN / (TN + FP);

    double fScore = 2 * precision * recall / (recall + precision);

    int totalInstances = TP + TN + FP + FN;
    int totalCorrectInstances = TP + TN;

    double accuracy = (double) totalCorrectInstances / totalInstances;

    cout << "Correctly Classified Instances         " << totalCorrectInstances << "               " << (accuracy * 100) << "% \n";
    cout << "Incorrectly Classified Instances        " << (totalInstances - totalCorrectInstances) << "               " << 100 - (accuracy * 100) << "% \n";
    cout << "Root mean squared error                  " << rmse << "\n";
    cout << "precision: " << precision << "\n";
    cout << "TN_rate: " << TN_rate << "\n";
    cout << "recall: " << recall << "\n";
    cout << "fScore: " << fScore << "\n";
    cout << "Total Number of Instances              " << totalInstances << "\n";

    cout << "=== Confusion Matrix ===\n";

    cout << "a   b   <-- classified as\n";
    cout << " " << TN << " " << FP << " |   a = 0\n";
    cout << " " << FN << " " <<  TP << " |   b = 1\n";

}