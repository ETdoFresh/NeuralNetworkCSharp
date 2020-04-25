using System.Collections.Generic;

namespace NeuralNetwork {
    public class Input : List<double>
    {
        public Input(params double[] inputs) => AddRange(inputs);

        public override string ToString()
        {
            var output = "";
            for (var i = 0; i < Count; i++)
                output += i == 0 ? $"{this[i]}" : $" {this[i]}";
            return output;
        }
    }
}