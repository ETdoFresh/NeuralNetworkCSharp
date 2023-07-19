using System;

namespace NeuralNetwork
{
    public static class RandomUtil
    {
        private static Random _random;
        private static int _seed;

        private static void InitRandom()
        {
            if (_random != null) return;
            _random = new Random();
        }
        
        public static void SetSeed(int seed)
        {
            if (_seed == seed) return;
            _seed = seed;
            _random = new Random(seed);
        }

        public static double Range(double min, double max)
        {
            InitRandom();
            return min + _random.NextDouble() * (max - min);
        }

        public static int Int()
        {
            InitRandom();
            return _random.Next();
        }

        public static bool Bool()
        {
            InitRandom();
            return _random.Next(0, 2) == 1;
        }

        public static double Double()
        {
            InitRandom();
            return _random.NextDouble();
        }
    }
}