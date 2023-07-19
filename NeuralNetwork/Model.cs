using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Model
    {
        public double LearningRate { get; set; } = 0.1;
        public LossFunction LossFunction { get; set; } = LossFunction.MeanSquaredError;
        public List<Layer> Layers { get; set; } = new List<Layer>();

        private List<Connection> _connections = new List<Connection>();
        private List<Neuron> _neurons = new List<Neuron>();

        public Model SetLearningRate(double learningRate)
        {
            LearningRate = learningRate;
            return this;
        }

        public Model SetLossFunction(LossFunction lossFunction)
        {
            LossFunction = lossFunction;
            return this;
        }

        public Model AddInputLayer(int numberOfNeurons)
        {
            var layer = new Layer("In", numberOfNeurons);
            Layers.Add(layer);
            _neurons.AddRange(layer.Neurons);
            return this;
        }

        public Model AddFullyConnectedLayer(int numberOfNeurons)
        {
            var layer = Layers.Count == 0
                ? new Layer("In", numberOfNeurons)
                : new FullyConnectedLayer(GetLayerId(), numberOfNeurons, Layers[Layers.Count - 1]);
            Layers.Add(layer);
            _neurons.AddRange(layer.Neurons);
            _connections.AddRange(layer.Connections);
            return this;
        }

        public Model AddFullyConnectedOutputLayer(int numberOfNeurons)
        {
            var layer = new FullyConnectedLayer("Out", numberOfNeurons, Layers[Layers.Count - 1]);
            Layers.Add(layer);
            _neurons.AddRange(layer.Neurons);
            _connections.AddRange(layer.Connections);
            return this;
        }

        private string GetLayerId()
        {
            var layerId = (char)('A' + Layers.Count - 1);
            return layerId.ToString();
        }

        public Model AddFullyConnectedLayers(params int[] neuronCounts)
        {
            foreach (var neuronCount in neuronCounts) 
                AddFullyConnectedLayer(neuronCount);

            return this;
        }

        public double[] Predict(double[] input)
        {
            var layer = Layers[0];
            for (var i = 0; i < layer.Neurons.Count; i++)
                layer.Neurons[i].A = input[i];

            for (var i = 1; i < Layers.Count; i++)
            {
                layer = Layers[i];
                foreach (var neuron in layer.Neurons)
                {
                    neuron.Z = 0;
                    foreach (var connection in neuron.Inputs)
                        neuron.Z += connection.Weight * connection.From.A;
                }
                
                foreach (var neuron in layer.Neurons)
                    neuron.A = neuron.Activate(neuron.Z);
            }
            
            var outputLayer = Layers[Layers.Count - 1];
            var output = new double[outputLayer.Neurons.Count];
            for (var i = 0; i < output.Length; i++)
                output[i] = outputLayer.Neurons[i].A;
            
            return output;
        }

        public void BackPropagate(double[] expectedOutputs)
        {
            var outputLayer = Layers[Layers.Count - 1];
            for (var i = 0; i < outputLayer.Neurons.Count; i++)
            {
                var neuron = outputLayer.Neurons[i];
                var expectedOutput = expectedOutputs[i];
                var actualOutput = neuron.A;
                neuron.Error = expectedOutput - actualOutput;
            }
            
            for (var i = Layers.Count - 2; i >= 0; i--)
            {
                var layer = Layers[i];
                foreach (var neuron in layer.Neurons)
                {
                    neuron.Error = 0;
                    foreach (var connection in neuron.Outputs)
                        neuron.Error += connection.To.Error * connection.Weight;
                }
            }
            
            foreach (var connection in _connections)
            {
                var error = connection.To.Error;
                var gradient = connection.To.Gradient;
                var previousOutputA = connection.From.A;
                connection.Weight += LearningRate * error * gradient * previousOutputA;
            }
            
            foreach (var neuron in _neurons)
            {
                var error = neuron.Error;
                var gradient = neuron.Gradient;
                neuron.Bias += LearningRate * error * gradient;
            }
        }
        
        public void Train(int epochs, int epochDisplayInterval, TrainingData trainingData)
        {
            var trainedEpochs = 0;
            var targetEpochs = trainedEpochs + epochs;
            DisplayOnConsole(1, trainedEpochs, trainingData);
            while (trainedEpochs < targetEpochs)
            {
                DisplayOnConsole(epochDisplayInterval, trainedEpochs, trainingData);
                foreach (var kvp in trainingData.Pairs)
                {
                    var input = kvp.Key;
                    var expectedOutput = kvp.Value;
                    Predict(input);
                    BackPropagate(expectedOutput);
                }

                trainedEpochs++;
            }
        }
        
        private void DisplayOnConsole(int epochDisplayInterval, int trainedEpochs, TrainingData trainingData)
        {
            var displayedEpoch = trainedEpochs + 1;
            var errorSum = 0.0;
            foreach (var kvp in trainingData.Pairs)
            {
                var input = kvp.Key;
                var expectedOutput = kvp.Value;
                var output = Predict(input);
                errorSum += ComputeDisplayError(expectedOutput, output);
                if (displayedEpoch % epochDisplayInterval == 0)
                {
                    var inputs = string.Join(", ", input.Select(i => i.ToString("0.00")).ToArray());
                    var expectedOutputs = string.Join(", ", expectedOutput.Select(i => i.ToString("0.00")).ToArray());
                    var outputs = string.Join(", ", output.Select(i => i.ToString("0.00")).ToArray());
                    Console.WriteLine(
                        $"Epoch: {displayedEpoch} Input: {inputs} Expected output: {expectedOutputs} Output: {outputs}");
                }
            }

            if (displayedEpoch % epochDisplayInterval == 0)
                Console.WriteLine("Epoch " + displayedEpoch + " error: " + errorSum);
        }
        
        private double ComputeDisplayError(double[] expectedOutput, double[] output)
        {
            var error = 0.0;
            for (var i = 0; i < expectedOutput.Length; i++)
            {
                error += Math.Abs(expectedOutput[i] - output[i]);
            }

            return error;
        }
        
        public void PrintInitialError(TrainingData trainingData)
        {
            var error = 0.0;
            foreach (var kvp in trainingData.Pairs)
            {
                var input = kvp.Key;
                var expectedOutput = kvp.Value;
                var output = Predict(input);
                error += ComputeDisplayError(expectedOutput, output);
            }

            error /= trainingData.Pairs.Count();
            Console.WriteLine("Initial error: " + error);
        }
    }

    public class TrainingData
    {
        private Dictionary<double[], double[]> pairs = new Dictionary<double[], double[]>();
        public double[][] Inputs => pairs.Keys.ToArray();
        public double[][] Outputs => pairs.Values.ToArray();
        public IEnumerable<KeyValuePair<double[], double[]>> Pairs => pairs;
        
        public int InputShape => Inputs.Max(x => x.Length);
        public int OutputShape => Outputs.Max(x => x.Length);

        public TrainingSetInputsTuple Input(params double[] inputs)
        {
            pairs.Add(inputs, new double[0]);
            return new TrainingSetInputsTuple {TrainingData = this, Inputs = inputs};
        }

        public class TrainingSetInputsTuple
        {
            public TrainingData TrainingData { get; set; } = new TrainingData();
            public double[] Inputs { get; set; } = new double[0];

            public TrainingData Output(params double[] outputs)
            {
                TrainingData.pairs[Inputs] = outputs;
                return TrainingData;
            }
        }
    }
}