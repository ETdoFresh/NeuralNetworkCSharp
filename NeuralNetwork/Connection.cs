namespace NeuralNetwork {
    public class Connection
    {
        public Neuron input;
        public Neuron output;
        public double weight;

        public Connection(Neuron input, Neuron output)
        {
            this.input = input;
            this.output = output;
        }

        public override string ToString()
            => $"{input} => {output} w: {weight}";
    }
}