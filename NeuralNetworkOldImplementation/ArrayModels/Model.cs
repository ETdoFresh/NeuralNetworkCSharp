using System.Collections.Generic;

namespace NeuralNetwork.ArrayModels
{
    /// <summary>
    /// Array classes not quite working yet
    /// </summary>
    public abstract class Model
    {
        public string ModelType { get; set; }
        public double LearningRate { get; set; } = 0.1;
        public string LossFunction { get; set; } = "Undefined/Mean Squared Error";
        public int TrainedEpochs { get; set; }
        public List<Layer> Layers { get; set; } = new List<Layer>();
        public List<Connection> Connections { get; set; } = new List<Connection>();

        protected TrainingData TrainingData { get; private set; } = new TrainingData();
        protected List<Neuron> Neurons { get; } = new List<Neuron>();

        public Model()
        {
            ModelType = GetType().Name;
        }

        public Model RebuildNeuronsAndConnections()
        {
            foreach (var layer in Layers)
            foreach (var neuron in layer.Neurons)
                Neurons.Add(neuron);
            
            foreach(var connection in Connections)
                connection.Model = this;

            foreach (var connection in Connections)
            foreach (var neuron in Neurons)
                if (neuron.Id == connection.From) neuron.Outputs.Add(connection);
                else if (neuron.Id == connection.To) neuron.Inputs.Add(connection);
            return this;
        }

        public virtual Model AddLayer(int numberOfNeurons)
        {
            var layer = new Layer(numberOfNeurons, Neurons.Count);
            Layers.Add(layer);
            Neurons.AddRange(layer.Neurons);
            return this;
        }

        public Model AddLayers(params int[] neuronCounts)
        {
            foreach (var neuronCount in neuronCounts)
            {
                AddLayer(neuronCount);
            }

            return this;
        }

        public abstract void Train(int epochs, int epochDisplayInterval);
        public abstract double[] Predict(double[] input);
        public abstract void PrintInitialError();

        public Model SetLearningRate(double learningRate)
        {
            LearningRate = learningRate;
            return this;
        }

        public Model SetTrainingData(TrainingData trainingData)
        {
            TrainingData = trainingData;
            return this;
        }

        public Model AddInputLayer() => AddLayer(TrainingData.InputShape);

        public Model AddOutputLayer() => AddLayer(TrainingData.OutputShape);

        public Neuron GetNeuron(int neuronIndex) => Neurons[neuronIndex];

        public int GetNeuronIndex(Neuron neuron) => Neurons.IndexOf(neuron);
    }
}