#include <iostream>
#include <cmath>


float activationFunction(float input)
{
    // Fast sigmoid function is non linier function with a fast and easy derivitive
    // f(x) = x / (1 + |x|)
    return input / (1 + std::abs(input));
}

float activationFunctionDerivitive(float input)
{
    float activatedValue = activationFunction(input);
    // Fast sigmoid function derivitive
    // f'(x) = f(x) * (1 - f(x))
    return activatedValue * (1 - activatedValue);
}

// We use mean squred error
float calculateCostFunction(float input, float expectedOutput)
{
    float error = 0.5 * std::pow(std::abs(expectedOutput - activationFunction(input)), 2);
    return error;
}

float calculateError(float input, float expectedOutput)
{
    float error = activationFunction(input) - expectedOutput;
    return error;
}

float calculateNewWeightForLastLayer(float input, float weight, float expectedOutput, float momentom, float learningRate)
{
     float error = calculateError(input, expectedOutput);
     float derivitiveValue = activationFunctionDerivitive(input);
     float gradiant = derivitiveValue * error;

     float deltaWeight = input * gradiant;
     float newWeight = (weight * momentom) - (deltaWeight  * learningRate);

     return newWeight;

}

float calculateNewWeightForLayer(float input, float weight, float previuesGradiant, float inputLayerActivated, float momentom, float learningRate)
{
     float gradiant = weight * previuesGradiant * activationFunctionDerivitive(input);

     float deltaWeight =  inputLayerActivated * gradiant;
     float newWeight = (weight * momentom) - (deltaWeight  * learningRate);

     return newWeight;
}

float feedForward(float input, float weight)
{
    return activationFunction(input) * weight;
}

float backPropagate(float input, float weight, float expectedOutput, float momentom, float learningRate)
{
     // this needs to be done for each layer of the neural network
     return calculateNewWeightForLastLayer(input, weight, expectedOutput, momentom, learningRate);
}


int main()
{
  std::cout << std::endl;
  std::cout << "working" << std::endl;

  float input = 25;
  float expectedOutput = 0.5;

  float weight = 0.5;

  float output = feedForward(input, weight);

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "input: " << input << std::endl;
  std::cout << "output: " << output << std::endl;
  std::cout << "expectedOutput: " << expectedOutput << std::endl;

  return 0;
}
