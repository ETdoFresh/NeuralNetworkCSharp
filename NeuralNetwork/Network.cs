using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Network
    {
        public Layer inputLayer;
        public List<Layer> hiddenLayers = new List<Layer>();
        public Layer outputLayer;
        public List<Neuron> neurons = new List<Neuron>();
        public List<Connection> connections = new List<Connection>();
        private double _learningRate = 0.1;

        public Network CreateInputLayerNeurons(int neuronCount)
        {
            inputLayer = new Layer(neuronCount);
            return this;
        }

        public Network CreateHiddenLayerNeurons(int neuronCount)
        {
            hiddenLayers.Add(new Layer(neuronCount));
            return this;
        }

        public Network CreateOutputLayerNeurons(int neuronCount)
        {
            outputLayer = new Layer(neuronCount);
            FinishNeuronConnections();
            return this;
        }

        public Network SetLearningRate(double learningRate)
        {
            _learningRate = learningRate;
            return this;
        }

        private void FinishNeuronConnections()
        {
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
                connection.weight = Random.Double();

            for (var layer = inputLayer; layer != null; layer = layer.next)
                foreach (var neuron in layer.neurons)
                    neuron.bias = Random.Double();
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

        public Output Predict(Input input)
        {
            for (var i = 0; i < input.Length; i++)
                inputLayer.neurons[i].inputZ = input[i];

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

            return new Output(outputLayer.neurons.Select(x => x.outputA).ToArray());
        }

        private void BackPropagate(Pair pair)
        {
            var input = pair.input;
            var output = pair.output;

            Predict(input);

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
                connection.weight += _learningRate * error * gradient * previousOutputA;
            }

            for (var layer = inputLayer; layer != null; layer = layer.next)
                foreach (var neuron in layer.neurons)
                {
                    var error = neuron.error;
                    var gradient = Sigmoid.Derivative(neuron.outputA);
                    neuron.bias += _learningRate * error * gradient;
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
                    Console.WriteLine($"Run #{i}: {expected.input} output: {predicted} error: {error}");
                }
            }
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