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
        
        public double[] ForwardPass(double[] inputs)
        {
            var layer = new Matrix(inputs);
            outputAs.Add(new Matrix(inputs));
            for (var i = 0; i < weights.Count; i++)
            {
                layer = weights[i].Multiply(layer);
                layer.Add(biases[i]);
                layer.Map(Sigmoid.Value);
                outputAs.Add(layer);
            }
            return layer.ToArray();
        }

        public void BackPropagate(Pair pair)
        {
            var inputs = pair.input.ToArray();
            var expectedOutputs = new Matrix(pair.output.ToArray());
            var outputs = new Matrix(ForwardPass(inputs));
            var outputErrors = Matrix.Subtract(expectedOutputs, outputs);
            
            var gradients = Matrix.Map(outputs, Sigmoid.Derivative);
            gradients.MultiplyElementWise(outputErrors);
            gradients.Multiply(learning_rate);

            var hidden = outputAs[1];
            var hiddenT = hidden.Transpose();
            var w1Deltas = Matrix.Multiply(gradients, hiddenT);

            biases[1].Add(gradients);
            weights[1].Add(w1Deltas);

            var w1T = weights[1].Transpose();
            var hiddenErrors = w1T.Multiply(outputErrors);
            var hiddenGradient = Matrix.Map(hidden, Sigmoid.Derivative);
            hiddenGradient.MultiplyElementWise(hiddenErrors);
            hiddenGradient.Multiply(learning_rate);
            var inputT = new Matrix(inputs).Transpose();
            var w0Deltas = hiddenGradient.Multiply(inputT);

            biases[0].Add(hiddenGradient);
            weights[0].Add(w0Deltas);
        }

        public override string ToString()
        {
            // var output = $"Input Layer\n";
            // output += "  Neurons\n";
            // foreach (var neuron in inputLayer.neurons)
            //     output += $"    {neuron.ToStringWithValues()}\n";
            // output += "  Connections\n";
            // foreach (var neuron in inputLayer.neurons)
            // foreach (var connection in neuron.outputs)
            //     output += $"    {connection}\n";
            //
            // for (var i = 0; i < hiddenLayers.Count; i++)
            // {
            //     output += $"Hidden Layer {i}\n";
            //     output += "  Neurons\n";
            //     foreach (var neuron in hiddenLayers[i].neurons)
            //         output += $"    {neuron.ToStringWithValues()}\n";
            //     output += "  Connections\n";
            //     foreach (var neuron in hiddenLayers[i].neurons)
            //     foreach (var connection in neuron.outputs)
            //         output += $"    {connection}\n";
            // }
            //
            // output += $"Output Layer\n";
            // output += "  Neurons\n";
            // foreach (var neuron in outputLayer.neurons)
            //     output += $"    {neuron.ToStringWithValues()}\n";
            // output += "  Connections\n";
            // foreach (var neuron in outputLayer.neurons)
            // foreach (var connection in neuron.outputs)
            //     output += $"    {connection}\n";
            //
            // return output;
            return base.ToString();
        }
    }
}