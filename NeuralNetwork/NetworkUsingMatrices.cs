using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class NetworkUsingMatrices
    {
        private List<Matrix> outputAs = new List<Matrix>();
        private List<Matrix> weights = new List<Matrix>();
        private List<Matrix> biases = new List<Matrix>();
        private double _learningRate = 0.1;
        private int _inputNeuronCount;

        public NetworkUsingMatrices CreateInputLayerNeurons(int neuronCount)
        {
            _inputNeuronCount = neuronCount;
            return this;
        }

        public NetworkUsingMatrices CreateHiddenLayerNeurons(int neuronCount)
        {
            weights.Add(weights.Count == 0
                ? new Matrix(neuronCount, _inputNeuronCount)
                : new Matrix(neuronCount, weights[weights.Count - 1].rows));
            biases.Add(new Matrix(neuronCount, 1));
            return this;
        }

        public NetworkUsingMatrices CreateOutputLayerNeurons(int neuronCount)
        {
            weights.Add(weights.Count == 0
                ? new Matrix(neuronCount, _inputNeuronCount)
                : new Matrix(neuronCount, weights[weights.Count - 1].rows));
            biases.Add(new Matrix(neuronCount, 1));
            
            foreach (var weight in weights)
                weight.Randomize();

            foreach (var bias in biases)
                bias.Randomize();
            
            return this;
        }

        public NetworkUsingMatrices SetLearningRate(double learningRate)
        {
            _learningRate = learningRate;
            return this;
        }

        public Output Predict(Input input) => Predict(new Matrix(input));

        private Output Predict(Matrix inputs)
        {
            var layer = inputs;
            outputAs.Clear();
            outputAs.Add(inputs);
            for (var i = 0; i < weights.Count; i++)
            {
                layer = Matrix.Multiply(weights[i], layer);
                layer.Add(biases[i]);
                layer.Map(Sigmoid.Value);
                outputAs.Add(layer);
            }

            return new Output(layer.ToArray());
        }

        private void BackPropagate(Pair pair)
        {
            var inputs = new Matrix(pair.input);

            Predict(inputs);

            var targets = new Matrix(pair.output);
            Matrix outputErrors = Matrix.Subtract(targets, outputAs[outputAs.Count - 1]);
            for (var i = outputAs.Count - 1; i > 0; i--)
            {
                var outputs = outputAs[i];
                if (i < outputAs.Count - 1)
                {
                    var weightT = Matrix.Transpose(weights[i]);
                    outputErrors = Matrix.Multiply(weightT, outputErrors);
                }

                var gradients = Matrix.Map(outputs, Sigmoid.Derivative);
                gradients.Multiply(outputErrors);
                gradients.Multiply(_learningRate);
                var previousOutputsT = Matrix.Transpose(outputAs[i - 1]);
                var weightDeltas = Matrix.Multiply(gradients, previousOutputsT);
                weights[i - 1].Add(weightDeltas);
                biases[i - 1].Add(gradients);
            }
        }

        public void Train(Batch batch, int iterations, int printLineRate = 0)
        {
            for (var i = 1; i <= iterations; i++)
            {
                var randomIndex = Random.Int(batch.Count);
                var pair = batch[randomIndex];
                BackPropagate(pair);
                if (printLineRate <= 0) continue;
                if (i % printLineRate != 0) continue;
                foreach (var expected in batch)
                {
                    var predicted = Predict(expected.input);
                    var error = Math.Abs(predicted[0] - expected.output[0]);
                    Console.WriteLine($"Run #{i}: Input: {expected.input} output: {predicted} error: {error}");
                }
            }
        }
        
        public void PrintInitialError(Batch batch)
        {
            var randomIndex = Random.Int(batch.Count);
            var expected = batch[randomIndex];
            var predictedOutput = Predict(expected.input);
            var error = Math.Abs(predictedOutput[0] - expected.output[0]);
            Console.WriteLine($"Initial: {expected.input} output: {predictedOutput} error: {error}");
        }
    }
}