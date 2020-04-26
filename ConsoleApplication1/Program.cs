﻿using System;
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

            var inputOutputPairs = new Batch(
                new Pair(new Input(0, 0), new Output(0)),
                new Pair(new Input(0, 1), new Output(1)),
                new Pair(new Input(1, 0), new Output(1)),
                new Pair(new Input(1, 1), new Output(0))
            );

            var randomIndex = random.Next(4);
            var mse = neuralNetwork.ComputeError(inputOutputPairs[randomIndex]);
            Console.WriteLine($"Starting Mean Squared Error: {mse}");

            for (int i = 0; i < 10000; i++)
            {
                randomIndex = random.Next(4);
                var pair = inputOutputPairs[randomIndex];
                neuralNetwork.BackPropagate(inputOutputPairs);
                var outputNeuron = neuralNetwork.outputLayer.neurons[0];
                mse = neuralNetwork.ComputeError(pair);
                Console.WriteLine(
                    $"Run #{i}: input: {pair.input[0]} {pair.input[1]} output: {outputNeuron.outputA} error: {mse}");
            }
        }
    }
}