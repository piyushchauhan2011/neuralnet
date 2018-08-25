#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>

#include "dataio.hpp"

using namespace std;

bool loadData(std::string filepath, vector<vector<double>> & dataset, size_t & numberOfInputs, size_t & numberOfOutputs) {
    bool result = false;
    ifstream fileStream(filepath); 

    if(fileStream) {
        string line;
        getline(fileStream, line);
        bool test = !line.empty();

        if(test) {
            std::istringstream headerStream(line);
            headerStream >> numberOfInputs;
            headerStream >> numberOfOutputs;
            int numberOfInstances;
            headerStream >> numberOfInstances;
            dataset.reserve(numberOfInstances);
            int counter = 0;
            while (test) {
                getline(fileStream, line);
                if (line.empty()) {
                    test = false;
                } else {
                    istringstream lineStream(line);
                    istream_iterator<double> begin(lineStream), eof;
                    vector<double> instance(begin, eof);
                    dataset.push_back(instance);
                }
                counter = counter + 1;
                test = counter < numberOfInstances;
            }
            result = true;
        }
    } else {
        cout << "file " << filepath << " not found." << endl;
    }
    return result;
}