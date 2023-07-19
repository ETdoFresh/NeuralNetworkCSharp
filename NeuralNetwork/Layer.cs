using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Layer
    {
        public string Id { get; set; }
        public List<Neuron> Neurons { get; set; } = new List<Neuron>();
        public List<Connection> Connections { get; set; } = new List<Connection>();

        public Layer(string id, int numberOfNeurons)
        {
            Id = id;
            for (var i = 0; i < numberOfNeurons; i++)
                Neurons.Add(new Neuron($"{Id}{i}"));
        }
    }
}