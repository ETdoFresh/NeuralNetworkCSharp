using System;

namespace NeuralNetwork
{
    public static class Activation
    {
        public static Func<double, double> GetActivationFunction(ActivationFunction activationFunction)
        {
            switch (activationFunction)
            {
                case ActivationFunction.Sigmoid:
                    return x => 1 / (1 + Math.Exp(-x));
                case ActivationFunction.Tanh:
                    return x => Math.Tanh(x);
                case ActivationFunction.ReLU:
                    return x => x > 0 ? x : 0;
                case ActivationFunction.LeakyReLU:
                    return x => x > 0 ? x : 0.01 * x;
                case ActivationFunction.Softmax:
                    return x => Math.Exp(x) / Math.Exp(x); // ??
                case ActivationFunction.None:
                    return x => x;
                default:
                    throw new ArgumentOutOfRangeException(nameof(activationFunction), activationFunction, null);
            }
        }

        public static Func<double, double> GetDerivativeFunction(ActivationFunction activationFunction)
        {
            switch (activationFunction)
            {
                case ActivationFunction.Sigmoid:
                    return x => x * (1 - x);
                case ActivationFunction.Tanh:
                    return x => 1 - Math.Pow(x, 2);
                case ActivationFunction.ReLU:
                    return x => x > 0 ? 1 : 0;
                case ActivationFunction.LeakyReLU:
                    return x => x > 0 ? 1 : 0.01;
                case ActivationFunction.Softmax:
                    return x => x * (1 - x); // ??
                case ActivationFunction.None:
                    return x => 1;
                default:
                    throw new ArgumentOutOfRangeException(nameof(activationFunction), activationFunction, null);
            }
        }
    }
}