using System;
using System.Collections.Generic;
using static NeuralNetwork.Activation;

namespace NeuralNetwork.ArrayModels
{
    /// <summary>
    /// Array classes not quite working yet
    /// </summary>
    public class Neuron
    {
        public int Id { get; set; }
        public string Activation { get; } = "Sigmoid";
        public double Bias { get; set; }

        internal List<Connection> Inputs { get; } = new List<Connection>();
        internal List<Connection> Outputs { get; } = new List<Connection>();
        internal double PreActivatedOutput { get; private set; }
        internal double PostActivatedOutput { get; set; }

        internal double InputZ
        {
            get => PreActivatedOutput;
            set => PreActivatedOutput = value;
        }

        internal double OutputA
        {
            get => PostActivatedOutput;
            set => PostActivatedOutput = value;
        }

        internal double Error { get; set; }
        internal double Gradient => DerivativeOfActivate(OutputA);

        internal Func<double, double> Activate => Sigmoid;
        internal Func<double, double> DerivativeOfActivate => SigmoidDerivative;

        public Neuron(int id)
        {
            Id = id;
            Bias = Randomizer.Double();
        }

        public override string ToString() =>
            $"Neuron {Id} Z: {InputZ:0.00} b: {Bias:0.00} A: {OutputA:0.00} Error: {Error:0.00}";
    }
}