using System;
using System.Linq;

namespace NeuralNetwork.ArrayModels
{
    public class SequentialModel : Model
    {
        /// <summary>
        /// Array classes not quite working yet
        /// </summary>
        public override double[] Predict(double[] input)
        {
            var layer = Layers[0];
            for (var i = 0; i < layer.Neurons.Count; i++)
            {
                layer.Neurons[i].OutputA = input[i];
            }

            for (var i = 1; i < Layers.Count; i++)
            {
                layer = Layers[i];
                foreach (var neuron in layer.Neurons)
                {
                    neuron.InputZ = 0;
                    foreach (var connection in neuron.Inputs)
                        neuron.InputZ += connection.Weight * connection.Input.OutputA;
                }
            }

            for (var i = 1; i < Layers.Count; i++)
            {
                layer = Layers[i];
                foreach (var neuron in layer.Neurons)
                {
                    neuron.OutputA = neuron.Activate(neuron.InputZ + neuron.Bias);
                }
            }

            return Layers.Last().Neurons.Select(n => n.OutputA).ToArray();
        }

        public override void Train(int epochs, int epochDisplayInterval)
        {
            var targetEpochs = TrainedEpochs + epochs;
            DisplayOnConsole(1);
            while (TrainedEpochs < targetEpochs)
            {
                DisplayOnConsole(epochDisplayInterval);
                BackPropagate();

                TrainedEpochs++;
            }
        }

        private void DisplayOnConsole(int epochDisplayInterval)
        {
            var displayedEpoch = TrainedEpochs + 1;
            var errorSum = 0.0;
            foreach (var (input, expectedOutput) in TrainingData.Pairs)
            {
                var output = Predict(input);
                errorSum += ComputeDisplayError(expectedOutput, output);
                if (displayedEpoch % epochDisplayInterval == 0)
                    Console.WriteLine("Epoch: " + displayedEpoch + " Input: " + string.Join(", ", input) +
                                      " Expected output: " + string.Join(", ", expectedOutput) + " Output: " +
                                      string.Join(", ", output));
            }

            if (displayedEpoch % epochDisplayInterval == 0)
                Console.WriteLine("Epoch " + displayedEpoch + " error: " + errorSum);
        }

        public void BackPropagate()
        {
            foreach (var (input, expectedOutput) in TrainingData.Pairs)
            {
                Predict(input);
                BackPropagate(expectedOutput);
            }
        }

        public void BackPropagate(double[] expectedOutputs)
        {
            var lastLayer = Layers.Last();
            for (var i = 0; i < lastLayer.Neurons.Count; i++)
            {
                var neuron = lastLayer.Neurons[i];
                var expectedOutput = expectedOutputs[i];
                var actualOutput = neuron.OutputA;
                neuron.Error = expectedOutput - actualOutput;
            }

            for (var i = Layers.Count - 2; i > 0; i--)
            {
                var layer = Layers[i];
                foreach (var neuron in layer.Neurons)
                {
                    neuron.Error = 0;
                    foreach (var output in neuron.Outputs)
                    {
                        var error = output.Output.Error;
                        var weight = output.Weight;
                        neuron.Error += error * weight;
                    }
                }
            }

            foreach (var connection in Connections)
            {
                var error = connection.Output.Error;
                var gradient = connection.Output.Gradient;
                var previousOutputA = connection.Input.OutputA;
                connection.Weight += LearningRate * error * gradient * previousOutputA;
            }
            
            foreach (var neuron in Neurons)
            {
                var error = neuron.Error;
                var gradient = neuron.Gradient;
                neuron.Bias += LearningRate * error * gradient;
            }
        }

        public override Model AddLayer(int neuronCount)
        {
            base.AddLayer(neuronCount);
            FullyConnectLastLayer();
            return this;
        }

        private void FullyConnectLastLayer()
        {
            if (Layers.Count < 2) return;
            var lastLayer = Layers.Last();
            var previousLayer = Layers[Layers.Count - 2];
            foreach (var previousLayerNeuron in previousLayer.Neurons)
            foreach (var nextLayerNeuron in lastLayer.Neurons)
            {
                var previousLayerNeuronIndex = GetNeuronIndex(previousLayerNeuron);
                var nextLayerNeuronIndex = GetNeuronIndex(nextLayerNeuron);
                var connection = new Connection(this, previousLayerNeuronIndex, nextLayerNeuronIndex);
                Connections.Add(connection);
                previousLayerNeuron.Outputs.Add(connection);
                nextLayerNeuron.Inputs.Add(connection);
            }
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

        public override void PrintInitialError()
        {
            var error = 0.0;
            foreach (var (input, expectedOutput) in TrainingData.Pairs)
            {
                var output = Predict(input);
                error += ComputeDisplayError(expectedOutput, output);
            }

            error /= TrainingData.Pairs.Count();
            Console.WriteLine("Initial error: " + error);
        }
    }
}