using System.Collections.Generic;

namespace NeuralNetwork.MatrixModels
{
    public class NeuralNetwork
    {
        private List<Matrix> activations = new();
        private List<Matrix> weights = new();
        private List<Matrix> biases = new();
        private int _firstLayerCount = -1;

        public void AddLayer(int neuronCount)
        {
            if (_firstLayerCount == -1)
            {
                _firstLayerCount = neuronCount;
            }
            else
            {
                var columns = weights.Count == 0 ? _firstLayerCount : weights[weights.Count - 1].Rows;
                weights.Add(new Matrix(neuronCount, columns));
                biases.Add(new Matrix(neuronCount, 1));
            }
        }

        public void Randomize()
        {
            foreach(var weight in weights)
                weight.Randomize();
            foreach(var bias in biases)
                bias.Randomize();
        }


    }
}