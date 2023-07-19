namespace NeuralNetwork
{
    public class FullyConnectedLayer : Layer
    {
        public FullyConnectedLayer(string id, int numberOfNeurons, Layer previousLayer) : base(id, numberOfNeurons)
        {
            foreach (var neuron in Neurons)
            foreach (var previousNeuron in previousLayer.Neurons)
                Connections.Add(new Connection(previousNeuron, neuron));
        }
    }
}