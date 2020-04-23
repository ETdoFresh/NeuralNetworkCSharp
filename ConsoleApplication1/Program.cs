using System;
using System.Collections.Generic;
using NeuralNetwork;

namespace ConsoleApplication1
{
    // Example Neural Network for XOR Learning

    // Neurons
    // Input Layer  Hidden Layer(s)  Output Layer
    // Input1       Hidden1          Output1
    // Input2       Hidden2

    // Connections
    // Input1-Hidden1, Input1-Hidden2, Input2-Hidden1, Input2-Hidden2
    // Hidden1-Output1, Hidden2-Output1

    public class Program
    {
        private static Random random = new Random();

        public static void Main(string[] args)
        {
            // Create Neural Network
            var learningRate = 0.1;
            var inputLayerNeurons = 2;
            var hiddenLayerNeurons = 2;
            var outputLayerNeurons = 1;
            var neuralNetwork = new Network(inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons);

            // Establish Inputs and Expected Outputs
            var inputs = new List<double[]>
            {
                new[] {0.0, 0.0}, new[] {0.0, 1.0}, new[] {1.0, 0.0}, new[] {1.0, 1.0}
            };
            var expectedOutputs = new List<double[]>
            {
                new[] {0.0}, new[] {1.0}, new[] {1.0}, new[] {0.0}
            };


            var randomIndex = random.Next(4);
            var mse = neuralNetwork.ComputeError(inputs[randomIndex], expectedOutputs[randomIndex]);
            Console.WriteLine($"Starting Mean Squared Error: {mse}");
            
            for (int i = 0; i < 10000; i++)
            {
                randomIndex = random.Next(4);
                // TODO: We are propagating via one pair at a time...
                // we may need to get the average error over a training set and then use that to correct weights...
                // Sum(outputLayer.mse) / n
                neuralNetwork.BackPropagate(inputs[randomIndex], expectedOutputs[randomIndex]);
                var outputNeuron = neuralNetwork.outputLayer.neurons[0];
                mse = neuralNetwork.ComputeError(inputs[randomIndex], expectedOutputs[randomIndex]);
                Console.WriteLine(
                    $"Run #{i}: input: {inputs[randomIndex][0]} {inputs[randomIndex][1]} output: {outputNeuron.activationOutput} error: {mse}");
            }
        }
    }
}