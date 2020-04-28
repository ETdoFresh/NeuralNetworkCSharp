using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class NetworkUsingMatrices
    {
        public List<Matrix> outputAs = new List<Matrix>();
        public List<Matrix> weights = new List<Matrix>();
        public List<Matrix> biases = new List<Matrix>();
        public double learning_rate = 0.1;

        public NetworkUsingMatrices(params int[] neuronCounts)
        {
            for (var i = 1; i < neuronCounts.Length; i++)
                weights.Add(new Matrix(neuronCounts[i], neuronCounts[i - 1]));

            for (var i = 1; i < neuronCounts.Length; i++)
                biases.Add(new Matrix(neuronCounts[i], 1));

            foreach (var weight in weights)
                weight.Randomize();

            foreach (var bias in biases)
                bias.Randomize();
        }

        public double[] Predict(double[] inputs) => Predict(new Matrix(inputs));

        public double[] Predict(Matrix inputs)
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

            return layer.ToArray();
        }

        public void BackPropagate(Pair pair)
        {
            var inputs = new Matrix(pair.input.ToArray());

            Predict(inputs);

            var targets = new Matrix(pair.output.ToArray());
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
                gradients.Multiply(learning_rate);
                var previousOutputsT = Matrix.Transpose(outputAs[i - 1]);
                var weightDeltas = Matrix.Multiply(gradients, previousOutputsT);
                weights[i - 1].Add(weightDeltas);
                biases[i - 1].Add(gradients);
            }
        }
    }
}