namespace NeuralNetwork {
    public class Pair
    {
        public Input input;
        public Output output;
        
        public Pair(Input input, Output output)
        {
            this.input = input;
            this.output = output;
        }

        public override string ToString() => $"Input: {input} Output: {output}";
    }
}