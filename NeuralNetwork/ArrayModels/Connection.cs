namespace NeuralNetwork.ArrayModels
{
    /// <summary>
    /// Array classes not quite working yet
    /// </summary>
    public class Connection
    {
        public int From { get; set; }
        public int To { get; set; }
        public double Weight { get; set; }
        
        internal Model Model { get; set; }
        internal Neuron Input => Model.GetNeuron(From);
        internal Neuron Output => Model.GetNeuron(To);

        public Connection() { }

        public Connection(Model model, int from, int to)
        {
            Model = model;
            From = from;
            To = to;
            Weight = Randomizer.Double();
        }
        
        public override string ToString() => $"{From} -> {To} w: {Weight:0.00}";
    }
}