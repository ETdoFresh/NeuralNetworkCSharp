using System;

namespace NeuralNetwork
{
    public static class Randomizer
    {
        private static readonly Random Random = new Random();
        
        public static double Range(double min, double max)
        {
            return min + Random.NextDouble() * (max - min);
        }

        public static int Int()
        {
            return Random.Next();
        }

        public static bool Bool()
        {
            return Random.Next(0, 2) == 1;
        }

        public static double Double()
        {
            return Random.NextDouble();
        }
    }
}