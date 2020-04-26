using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Network
    {
        public static Random random = new Random();

        public Layer inputLayer;
        public List<Layer> hiddenLayers = new List<Layer>();
        public Layer outputLayer;
        public List<Neuron> neurons = new List<Neuron>();
        public List<Connection> connections = new List<Connection>();

        public Network(params int[] neuronCounts)
        {
            if (neuronCounts.Length < 2)
                throw new ArgumentException("Not enough counts, need at least two for Input and Output layers");

            // Create Layers
            inputLayer = new Layer(neuronCounts[0]);
            for (var i = 1; i < neuronCounts.Length - 1; i++)
                hiddenLayers.Add(new Layer(neuronCounts[i]));
            outputLayer = new Layer(neuronCounts[neuronCounts.Length - 1]);

            // Get All Neurons
            foreach (var neuron in inputLayer.neurons)
                neurons.Add(neuron);
            foreach (var layer in hiddenLayers)
            foreach (var neuron in layer.neurons)
                neurons.Add(neuron);
            foreach (var neuron in outputLayer.neurons)
                neurons.Add(neuron);

            // Connect Layer Together
            if (hiddenLayers.Count > 0)
            {
                inputLayer.SetNext(hiddenLayers[0]);
                for (var i = 1; i < hiddenLayers.Count; i++)
                    hiddenLayers[i - 1].SetNext(hiddenLayers[i]);
                hiddenLayers[hiddenLayers.Count - 1].SetNext(outputLayer);
            }
            else
                inputLayer.SetNext(outputLayer);

            // Fully Connect Layers
            for (var layer = inputLayer.next; layer != null; layer = layer.next)
                FullyConnect(layer.previous, layer);

            // Randomize Weights and Biases
            foreach (var connection in connections)
                connection.weight = random.NextDouble();

            inputLayer.bias = random.NextDouble();
            foreach (var layer in hiddenLayers)
                layer.bias = random.NextDouble();
            outputLayer.bias = random.NextDouble();
        }

        private void FullyConnect(Layer a, Layer b)
        {
            for (var i = 0; i < a.neurons.Count; i++)
            for (var j = 0; j < b.neurons.Count; j++)
            {
                var input = a.neurons[i];
                var output = b.neurons[j];
                var connection = new Connection(input, output);
                input.outputs.Add(connection);
                output.inputs.Add(connection);
                connections.Add(connection);
            }
        }

        public void SetInputs(double[] inputs)
        {
            for (int i = 0; i < inputs.Length; i++)
                inputLayer.neurons[i].inputZ = inputs[i];
        }

        public void ForwardPass(double[] inputs)
        {
            SetInputs(inputs);
            foreach (var inputNeuron in inputLayer.neurons)
                inputNeuron.outputA = Sigmoid.Value(inputNeuron.inputZ + inputLayer.bias);

            for (var layer = inputLayer.next; layer != null; layer = layer.next)
                foreach (var neuron in layer.neurons)
                {
                    foreach (var connection in neuron.inputs)
                        neuron.inputZ += connection.input.outputA * connection.weight;

                    neuron.outputA = Sigmoid.Value(neuron.inputZ + layer.bias);
                }
        }

        // public void BackPropagate(Batch batch)
        // {
        //     var learning_rate = 0.1;
        //     for (var layer = outputLayer; layer.previous != null; layer = layer.previous)
        //         foreach (var neuron in layer.neurons)
        //         foreach (var connection in neuron.inputs)
        //         {
        //             var deltaBiasSum = 0.0;
        //             var deltaWeightSum = 0.0;
        //             foreach (var pair in batch)
        //             {
        //                 ForwardPass(pair.input.ToArray());
        //                 for (var i = 0; i < pair.output.Count; i++)
        //                     outputLayer.neurons[i].expectedActivationOutput = pair.output[i];
        //
        //                 // dLoss/dWeight = (dLoss/dCurrentActivation)(dCurrentActivation/dCurrentInput)(dCurrentInput/dWeight)
        //                 // dLoss/dWeight = 2(a-y)(g'(z))(prevA)
        //                 var currentActivation = neuron.outputA;
        //                 var expectedActivation = neuron.expectedActivationOutput;
        //                 var derivativeInput = Sigmoid.Derivative(neuron.inputZ);
        //                 var previousActivation = connection.input.outputA;
        //
        //                 var delta = 2 * (currentActivation - expectedActivation);
        //                 delta *= derivativeInput;
        //
        //                 deltaBiasSum += delta;
        //
        //                 delta *= previousActivation;
        //                 deltaWeightSum += delta;
        //             }
        //
        //             neuron.bias += learning_rate * deltaBiasSum / batch.Count;
        //             connection.weight += learning_rate * deltaWeightSum / batch.Count;
        //         }
        // }

        public void BackPropagate(Pair pair)
        {
            var input = pair.input.ToArray();
            var output = pair.output.ToArray();
            var learning_rate = 0.1;

            ForwardPass(input);
            
            for (var i = 0; i < outputLayer.neurons.Count; i++)
            {
                var neuron = outputLayer.neurons[i];
                var outputA = neuron.outputA;
                var expectedOutput = output[i];
                var dCdA = 2 * (outputA - expectedOutput);
                neuron.costCorrection = dCdA;
            }

            for (var layer = outputLayer.previous; layer != inputLayer; layer = layer.previous)
            {
                foreach (var neuron in layer.neurons)
                {
                    neuron.costCorrection = 0;
                    foreach (var connection in neuron.outputs)
                    {
                        var dCdA = connection.output.costCorrection;
                        var dAdZ = Sigmoid.Derivative(connection.output.outputA);
                        var dZdW = connection.weight;
                        neuron.costCorrection += dCdA * dAdZ * dZdW;
                    }
                }
            }

            foreach (var connection in connections)
            {
                var dCdA = connection.output.costCorrection;
                var dAdZ = Sigmoid.Derivative(connection.output.outputA);
                var dZdA = connection.input.outputA;
                connection.weight += dCdA * dAdZ * dZdA * learning_rate;
            }

            foreach(var hiddenLayer in hiddenLayers)
            foreach (var neuron in hiddenLayer.neurons)
            {
                var dCdA = neuron.costCorrection;
                var dAdZ = Sigmoid.Derivative(neuron.outputA);
                var dZdB = 1;
                hiddenLayer.bias += dCdA * dAdZ * dZdB * learning_rate;
            }
        }

        public double ComputeError(Pair pair)
        {
            var errors = GetErrors(pair.output.ToArray());
            return GeMeanSquaredError(errors);
        }

        private double[] GetErrors(double[] expectedOutputs)
        {
            double[] errors = new double[expectedOutputs.Length];
            for (var i = 0; i < expectedOutputs.Length; i++)
            {
                var expectedOutput = expectedOutputs[i];
                var outputNeuron = outputLayer.neurons[i];
                var outputValue = outputNeuron.outputA;
                errors[i] = expectedOutput - outputValue;
            }

            return errors;
        }

        private double GeMeanSquaredError(double[] errors)
        {
            var meanSquaredError = 0.0;
            foreach (var error in errors)
                meanSquaredError += error * error;
            meanSquaredError /= errors.Length;
            return meanSquaredError;
        }

        public override string ToString()
        {
            var output = $"Input Layer b: {inputLayer.bias}\n";
            output += "  Neurons\n";
            foreach (var neuron in inputLayer.neurons)
                output += $"    {neuron.ToStringWithValues()}\n";
            output += "  Connections\n";
            foreach (var neuron in inputLayer.neurons)
            foreach (var connection in neuron.outputs)
                output += $"    {connection}\n";

            for (var i = 0; i < hiddenLayers.Count; i++)
            {
                output += $"Hidden Layer {i} b: {hiddenLayers[i].bias}\n";
                output += "  Neurons\n";
                foreach (var neuron in hiddenLayers[i].neurons)
                    output += $"    {neuron.ToStringWithValues()}\n";
                output += "  Connections\n";
                foreach (var neuron in hiddenLayers[i].neurons)
                foreach (var connection in neuron.outputs)
                    output += $"    {connection}\n";
            }

            output += $"Output Layer b: {outputLayer.bias}\n";
            output += "  Neurons\n";
            foreach (var neuron in outputLayer.neurons)
                output += $"    {neuron.ToStringWithValues()}\n";
            output += "  Connections\n";
            foreach (var neuron in outputLayer.neurons)
            foreach (var connection in neuron.outputs)
                output += $"    {connection}\n";

            return output;
        }
    }
}