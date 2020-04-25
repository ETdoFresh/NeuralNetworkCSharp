using System.Collections.Generic;

namespace NeuralNetwork {
    public class Neuron
    {
        public static int idCounter = 0;

        public int id;
        public double inputZ;
        public double outputA;
        public double costCorrection;
        public List<Connection> inputs = new List<Connection>();
        public List<Connection> outputs = new List<Connection>();

        public Neuron()
        {
            id = idCounter++;
        }
        
        public override string ToString() => $"Neuron {id}";

        public string ToStringWithValues() => $"{this} z: {inputZ} a: {outputA}";
    }
}