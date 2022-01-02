using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Batch : List<Pair>
    {
        public Batch Input(params double[] inputValues)
        {
            var input = new Input(inputValues);
            var pair = new Pair(input, new Output());
            Add(pair);
            return this;
        }

        public Batch Output(params double[] outputValues)
        {
            var output = new Output(outputValues);
            var pair = Count > 0 ? this[Count - 1] : new Pair(null, null);
            pair.output = output;
            if (Count == 0) Add(pair);
            return this;
        }
    }
}