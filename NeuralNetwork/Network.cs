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
        public double learning_rate = 0.1;

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

            for (var layer = inputLayer; layer != null; layer = layer.next)
                foreach (var neuron in layer.neurons)
                    neuron.bias = random.NextDouble();
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
                inputNeuron.outputA = Sigmoid.Value(inputNeuron.inputZ + inputNeuron.bias);

            for (var layer = inputLayer.next; layer != null; layer = layer.next)
                foreach (var neuron in layer.neurons)
                {
                    neuron.inputZ = 0;
                    foreach (var connection in neuron.inputs)
                        neuron.inputZ += connection.input.outputA * connection.weight;

                    neuron.outputA = Sigmoid.Value(neuron.inputZ + neuron.bias);
                }
        }

        public void BackPropagate(Pair pair)
        {
            var input = pair.input.ToArray();
            var output = pair.output.ToArray();

            ForwardPass(input);

            for (var i = 0; i < outputLayer.neurons.Count; i++)
            {
                var neuron = outputLayer.neurons[i];
                var outputA = neuron.outputA;
                var expectedOutput = output[i];
                neuron.error = expectedOutput - outputA;
            }

            for (var layer = outputLayer.previous; layer != null; layer = layer.previous)
            {
                foreach (var neuron in layer.neurons)
                {
                    neuron.error = 0;
                    foreach (var connection in neuron.outputs)
                    {
                        var error = connection.output.error;
                        var weight = connection.weight;
                        neuron.error += error * weight;
                    }
                }
            }

            foreach (var connection in connections)
            {
                var error = connection.output.error;
                var gradient = Sigmoid.Derivative(connection.output.outputA);
                var previousOutputA = connection.input.outputA;
                connection.weight += learning_rate * error * gradient * previousOutputA;
            }

            for (var layer = inputLayer; layer != null; layer = layer.next)
                foreach (var neuron in layer.neurons)
                {
                    var error = neuron.error;
                    var gradient = Sigmoid.Derivative(neuron.outputA);
                    neuron.bias += learning_rate * error * gradient;
                }
        }

        public void BackPropagate(Batch batch)
        {
            var learning_rate = 0.01;
            foreach (var neuron in neurons)
                neuron.error = 0;

            foreach (var connection in connections)
                connection.costCorrection = 0;

            foreach (var pair in batch)
            {
                var input = pair.input.ToArray();
                var output = pair.output.ToArray();

                ForwardPass(input);

                for (var i = 0; i < outputLayer.neurons.Count; i++)
                {
                    var neuron = outputLayer.neurons[i];
                    var outputA = neuron.outputA;
                    var expectedOutput = output[i];
                    var error = expectedOutput - outputA;
                    var dCdA = 2 * error;
                    neuron.error += dCdA / batch.Count;
                }

                for (var layer = outputLayer.previous; layer != null; layer = layer.previous)
                {
                    foreach (var neuron in layer.neurons)
                    {
                        foreach (var connection in neuron.outputs)
                        {
                            var dCdA = connection.output.error;
                            var dAdZ = Sigmoid.Derivative(connection.output.outputA);
                            var dZdA = connection.weight;
                            var dZdW = connection.input.outputA;
                            neuron.error += dCdA * dAdZ * dZdA / batch.Count;
                            connection.costCorrection += dCdA * dAdZ * dZdW / batch.Count;
                        }
                    }
                }
            }

            foreach (var connection in connections)
            {
                connection.weight += connection.costCorrection * learning_rate;
            }

            for (var layer = inputLayer; layer != null; layer = layer.next)
                foreach (var neuron in layer.neurons)
                {
                    var dCdA = neuron.error;
                    var dAdZ = Sigmoid.Derivative(neuron.outputA);
                    var dZdB = 1;
                    neuron.bias += dCdA * dAdZ * dZdB * learning_rate;
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
            var output = $"Input Layer\n";
            output += "  Neurons\n";
            foreach (var neuron in inputLayer.neurons)
                output += $"    {neuron.ToStringWithValues()}\n";
            output += "  Connections\n";
            foreach (var neuron in inputLayer.neurons)
            foreach (var connection in neuron.outputs)
                output += $"    {connection}\n";

            for (var i = 0; i < hiddenLayers.Count; i++)
            {
                output += $"Hidden Layer {i}\n";
                output += "  Neurons\n";
                foreach (var neuron in hiddenLayers[i].neurons)
                    output += $"    {neuron.ToStringWithValues()}\n";
                output += "  Connections\n";
                foreach (var neuron in hiddenLayers[i].neurons)
                foreach (var connection in neuron.outputs)
                    output += $"    {connection}\n";
            }

            output += $"Output Layer\n";
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