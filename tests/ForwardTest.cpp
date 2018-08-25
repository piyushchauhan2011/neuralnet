#include "ann.hpp"

#include "gtest/gtest.h"

class ForwardTest : public ::testing::Test {

};

TEST_F(ForwardTest, ForwardTest_example) {

    std::vector<std::vector<std::vector<double>>> initialWeights {

        {
            {.3, -.1, .15, .25, 0.0}, {.1, .25, .25, -.2, 0.0}, {-.5, .3, .15, -.1, 0.0}
        },
        {
            {.3, .4, -.2, 0.0}, {.1, -.12, .2, 0.0}
        }
 
    };

    NeuralNet net(initialWeights);

    std::vector<double> output = net.forward({.1, .5, .2, .8});

    EXPECT_NEAR(0.566, output[0], 0.0001);

}