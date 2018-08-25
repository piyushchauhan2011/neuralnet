# neuralnet
It is an implementation of the Multilayer Perceptron in C++11. The model is based on the one described in the book Neural Networks: a comprehesive foundation, by Simon Haykin. The objective of this implementation is just for educational purpose for present the basis of the most popular artificial neural network.

## The implementation

Some notes about the implementation:

- The topology is represented by a vector of integer for denote the number of neurons in each layer. It is used a representation of the weight matrix as well. 

- The network is memoryless. The Neuron's hold value is only present during the forward execution.

- Neither momento nor learning rate decay is implemented. Feel free to implement it for yourself as exercise as decribed in the book.

- Sigmoid is used as activation function as described in the book.

- It is applied just one stop condition. The backpropagation finishes when the epoch reaches the maximum value. Feel free to implement another strategies like the mean squared error stability, as also decribed in the book.

- The network just supports float points (implemented as double float points) and outputs just the score. Thus could be used both for classification as for regression problems.

- It is provided a single dataset in two formats: plain data e arff. The arff data is used by the software Weka and is provided just for verification purposes.

- The main.cpp file conduces the training and validation steps of the network. So far, the validation is done by simples hold out with 66% for training. More complex cross validation approaches are desired and leave for the view as practice.

- The training data is shuffled for each epoch, as indicated in the book, ir order to explore different regions of the error surfice and avoid bad local optimas. Of course it slow down the training execution and as suggestion it can be disabled in runtime based on a criteria of norma of the gradient or RMSE instability (TODO).

- The implementation is in C++ std 11 and so far the code was compliled using GCC 5.4

- It is provided a CMAKE file configuration to be used with MAKE in order to build the sources. The defaul CMAKE configuration is Relase. This configuration cab be changed by the CMAKE_BUIL_TYPE property.

- The cmake file is blunded with a BUILD_TESTS flag for tests purposes. Once cmake is called with -DBUILD_TESTS=ON, make command will try do clone localy the googletest framework and then compile it and link.

- The current version just provide a single Test Case for the ```forward``` public method. It is strongly recommended to implement more test case to explore more paths into ```forward``` and for the other features like constructors and the ```train``` method.


