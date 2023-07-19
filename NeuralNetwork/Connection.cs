namespace NeuralNetwork
{
    public class Connection
    {
        public Neuron From { get; set; }
        public Neuron To { get; set; }
        public double Weight { get; set; }

        private static double FromWeightRange { get; set; } = -1.00;
        private static double ToWeightRange { get; set; } = 1.00;

        public Connection(Neuron from, Neuron to)
        {
            From = from;
            To = to;
            Weight = RandomUtil.Range(FromWeightRange, ToWeightRange);
            to.Inputs.Add(this);
            from.Outputs.Add(this);
        }

        public static void SetRandomWeightRange(double fromWeightRange, double toWeightRange)
        {
            FromWeightRange = fromWeightRange;
            ToWeightRange = toWeightRange;
        }

        public override string ToString() => $"{From} -> {To} w: {Weight:0.00}";
    }
}