#ifndef DATA_IO_H
#define DATA_IO_H

#include <vector>
#include <string>

bool loadData(std::string filepath, std::vector<std::vector<double>> & dataset, size_t & numberOfInputs, size_t & numberOfOutputs);

#endif