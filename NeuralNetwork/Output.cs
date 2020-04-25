using System.Collections.Generic;

namespace NeuralNetwork {
    public class Output : List<double>
    {
        public Output(params double[] outputs) => AddRange(outputs);
        
        public override string ToString()
        {
            var output = "";
            for (var i = 0; i < Count; i++)
                output += i == 0 ? $"{this[i]}" : $" {this[i]}";
            return output;
        }
    }
}