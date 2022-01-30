using System;

namespace NeuralNetwork
{
    public static class Activation
    {
        public static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public static double SigmoidDerivative(double sigmoid)
        {
            //return Sigmoid(x) * (1 - Sigmoid(x));
            return sigmoid * (1 - sigmoid);
        }
        
        public static double Tanh(double arg)
        {
            return Math.Tanh(arg);
        }
        
        public static double TanhDerivative(double arg)
        {
            return 1 - Math.Pow(Tanh(arg), 2);
        }
        
        public static double ReLU(double arg)
        {
            return Math.Max(0, arg);
        }
        
        public static double ReLUDerivative(double arg)
        {
            return arg > 0 ? 1 : 0;
        }
    }
}