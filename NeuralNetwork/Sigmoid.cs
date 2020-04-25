using System;

namespace NeuralNetwork {
    public static class Sigmoid
    {
        public static double Value(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }

        public static double Derivative(double sigmoid)
        {
            return sigmoid * (1 - sigmoid);
        }
    }
}