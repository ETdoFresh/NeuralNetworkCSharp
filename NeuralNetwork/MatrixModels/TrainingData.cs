using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.MatrixModels
{
    public class TrainingData
    {
        private Dictionary<Matrix, Matrix> pairs = new();
        public Matrix[] Inputs => pairs.Keys.ToArray();
        public Matrix[] Outputs => pairs.Values.ToArray();
        public IEnumerable<KeyValuePair<Matrix, Matrix>> Pairs => pairs;

        public int InputShape => Inputs.Max(x => x.Length);
        public int OutputShape => Outputs.Max(x => x.Length);
        
        public TrainingSetInputsTuple Input(params double[] inputArray)
        {
            var inputs = new Matrix(inputArray);
            pairs.Add(inputs, Matrix.Empty);
            return new TrainingSetInputsTuple {TrainingData = this, Inputs = inputs};
        }

        public class TrainingSetInputsTuple
        {
            public TrainingData TrainingData { get; init; } = new();
            public Matrix Inputs { get; init; } = Matrix.Empty;

            public TrainingData Output(params double[] outputArray)
            {
                TrainingData.pairs[Inputs] = new Matrix(outputArray);
                return TrainingData;
            }
        }
    }
}