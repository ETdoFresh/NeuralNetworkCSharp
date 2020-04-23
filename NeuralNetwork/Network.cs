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
            outputLayer = new Layer(neuronCounts[neuronCounts.Length - 1]);
            for (var i = 1; i < neuronCounts.Length - 1; i++)
                hiddenLayers.Add(new Layer(neuronCounts[i]));

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

            foreach (var neuron in neurons)
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
                inputLayer.neurons[i].inputValue = inputs[i];
        }

        public void ForwardPass(double[] inputs)
        {
            SetInputs(inputs);
            for (var layer = inputLayer; layer != null; layer = layer.next)
            {
                foreach (var neuron in layer.neurons)
                {
                    if (layer != inputLayer)
                        neuron.inputValue = 0;

                    foreach (var connection in neuron.inputs)
                        neuron.inputValue += connection.input.activationOutput * connection.weight;

                    neuron.activationOutput = Sigmoid.Value(neuron.inputValue + neuron.bias);
                }
            }
        }

        public void BackPropagate(double[] inputs, double[] expectedOutputs)
        {
            ForwardPass(inputs);
            for (var i = 0; i < expectedOutputs.Length; i++)
                outputLayer.neurons[i].expectedActivationOutput = expectedOutputs[i];

            var learning_rate = 0.1;
            for (var layer = outputLayer; layer.previous != null; layer = layer.previous)
                foreach (var neuron in layer.neurons)
                foreach (var connection in neuron.inputs)
                {
                    // dLoss/dWeight = (dLoss/dCurrentActivation)(dCurrentActivation/dCurrentInput)(dCurrentInput/dWeight)
                    // dLoss/dWeight = 2(a-y)(g'(z))(prevA)
                    var currentActivation = neuron.activationOutput;
                    var expectedActivation = neuron.expectedActivationOutput;
                    var derivativeInput = Sigmoid.Derivative(neuron.inputValue);
                    var previousActivation = connection.input.activationOutput;

                    var delta = 2 * (expectedActivation - currentActivation);
                    delta *= derivativeInput;
                    neuron.bias += learning_rate * delta;

                    var deltaWeight = delta * previousActivation;
                    connection.weight += learning_rate * deltaWeight;
                }
        }

        public double ComputeError(double[] inputs, double[] expectedOutputs)
        {
            ForwardPass(inputs);
            GetErrors(expectedOutputs);
            GetOutputLayerError();
            return outputLayer.meanSquaredError;
        }

        private void GetErrors(double[] expectedOutputs)
        {
            for (var i = 0; i < expectedOutputs.Length; i++)
            {
                var expectedOutput = expectedOutputs[i];
                var outputNeuron = outputLayer.neurons[i];
                var outputValue = outputNeuron.activationOutput;
                var error = outputValue - expectedOutput;
                outputNeuron.error = error;
            }
        }

        public void GetOutputLayerError()
        {
            outputLayer.meanSquaredError = 0;
            foreach (var neuron in outputLayer.neurons)
                outputLayer.meanSquaredError += neuron.error * neuron.error;
            outputLayer.meanSquaredError /= outputLayer.neurons.Count;
        }
    }
}