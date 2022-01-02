namespace NeuralNetwork
{
    public static class Random
    {
        private static System.Random _random = new System.Random();

        public static void SetSeed(int seed) => _random = new System.Random(seed);
        public static int Int() => _random.Next();
        public static int Int(int maxValue) => _random.Next(maxValue);
        public static int Int(int minValue, int maxValue) => _random.Next(minValue, maxValue);
        public static double Double() => _random.NextDouble();
        public static void Bytes(byte[] buffer) => _random.NextBytes(buffer);
    }
}