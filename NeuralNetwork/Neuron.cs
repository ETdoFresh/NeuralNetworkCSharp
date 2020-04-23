using System.Collections.Generic;

namespace NeuralNetwork {
    public class Neuron
    {
        public double inputValue;
        public double activationOutput;
        public double bias;
        public double error;
        public List<Connection> inputs = new List<Connection>();
        public List<Connection> outputs = new List<Connection>();
        public double expectedActivationOutput;

        public override string ToString() => $"Neuron z: {inputValue} a: {activationOutput}";
    }
}