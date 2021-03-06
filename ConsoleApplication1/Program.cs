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
            var inputLayerNeurons = 2;
            var hiddenLayerNeurons = 4;
            var outputLayerNeurons = 1;
            var neuralNetwork = new NetworkUsingMatrices(inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons);
            neuralNetwork.learning_rate = 0.1;
            
            var inputOutputPairs = new Batch(
                new Pair(new Input(0, 0), new Output(0)),
                new Pair(new Input(0, 1), new Output(1)),
                new Pair(new Input(1, 0), new Output(1)),
                new Pair(new Input(1, 1), new Output(0))
            );

            {
                var randomIndex = random.Next(4);
                var pair = inputOutputPairs[randomIndex];
                var output = neuralNetwork.Predict(pair.input.ToArray());
                var error = Math.Abs(output[0] - pair.output[0]);
                Console.WriteLine(
                    $"Initial: {pair.input[0]} {pair.input[1]} output: {output[0]} error: {error}");
            }

            for (var i = 0; i < 50000; i++)
            {
                var randomIndex = random.Next(4);
                var pair = inputOutputPairs[randomIndex];
                neuralNetwork.BackPropagate(pair);
                if (i % 100 == 0)
                    foreach (var p in inputOutputPairs)
                    {
                        var output = neuralNetwork.Predict(p.input.ToArray());
                        var error = Math.Abs(output[0] - p.output[0]);
                        Console.WriteLine(
                            $"Run #{i}: {p.input[0]} {p.input[1]} output: {output[0]} error: {error}");
                    }
            }

            // for (var i = 0; i < 100; i++)
            // {
            //     var randomIndex = random.Next(4);
            //     var pair = inputOutputPairs[randomIndex];
            //     neuralNetwork.BackPropagate(pair);
            //     var outputNeuron = neuralNetwork.outputLayer.neurons[0];
            //     var error = Math.Abs(outputNeuron.outputA - pair.output[0]);
            //     Console.WriteLine(
            //         $"Run #{i}: input: {pair.input[0]} {pair.input[1]} output: {outputNeuron.outputA} error: {error}");
            // }
        }
    }
}