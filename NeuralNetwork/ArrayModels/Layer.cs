using System.Collections.Generic;

namespace NeuralNetwork.ArrayModels
{
    /// <summary>
    /// Array classes not quite working yet
    /// </summary>
    public class Layer
    {
        public string LayerType { get; set; } = "Dense";
        public List<Neuron> Neurons { get; set; } = new();

        public Layer() { }

        public Layer(int neurons, int startIndex)
        {
            for (var i = 0; i < neurons; i++)
            {
                Neurons.Add(new Neuron(startIndex + i));
            }
        }
    }
}