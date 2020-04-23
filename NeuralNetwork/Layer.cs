using System.Collections.Generic;

namespace NeuralNetwork {
    public class Layer
    {
        public Layer previous;
        public Layer next;
        public List<Neuron> neurons = new List<Neuron>();
        public double meanSquaredError;

        public Layer(int neuronCount)
        {
            for (var i = 0; i < neuronCount; i++)
                neurons.Add(new Neuron());
        }

        public void SetNext(Layer next)
        {
            this.next = next;
            next.previous = this;
        }
    }
}