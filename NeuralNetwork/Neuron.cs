using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Neuron
    {
        public string Id { get; private set; }
        public ActivationFunction ActivationFunction { get; private set; }
        public double Bias { get; set; }

        public double Z
        {
            get => _preActivatedOutput;
            set => _preActivatedOutput = value;
        }

        public double A
        {
            get => _postActivatedOutput;
            set => _postActivatedOutput = value;
        }

        public double Error { get; set; }
        public double Gradient => DerivativeOfActivate(_postActivatedOutput);

        public Func<double, double> Activate => Activation.GetActivationFunction(ActivationFunction);
        private Func<double, double> DerivativeOfActivate => Activation.GetDerivativeFunction(ActivationFunction);
        public List<Connection> Inputs { get; set; } = new List<Connection>();
        public List<Connection> Outputs { get; set; } = new List<Connection>();

        private double _preActivatedOutput;
        private double _postActivatedOutput;

        private static double FromBiasRange { get; set; } = -1.00;
        private static double ToBiasRange { get; set; } = 1.00;

        public Neuron(string id, ActivationFunction activationFunction = ActivationFunction.LeakyReLU)
        {
            Id = id;
            ActivationFunction = activationFunction;
            Bias = RandomUtil.Range(FromBiasRange, ToBiasRange);
        }

        public static void SetRandomBiasRange(double fromBiasRange, double toBiasRange)
        {
            FromBiasRange = fromBiasRange;
            ToBiasRange = toBiasRange;
        }

        public override string ToString() => $"Neuron {Id} Z: {Z:0.00} b: {Bias:0.00} A: {A:0.00} Error: {Error:0.00}";
    }
}