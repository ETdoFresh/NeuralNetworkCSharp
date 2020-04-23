using System;

namespace NeuralNetwork {
    public static class Sigmoid
    {
        public static double Value(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }

        public static double Derivative(double value)
        {
            return value * (1 - Value(value));
        }
    }
}