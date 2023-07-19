using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.ArrayModels
{
    /// <summary>
    /// Array classes not quite working yet
    /// </summary>
    public class TrainingData
    {
        private Dictionary<double[], double[]> pairs = new Dictionary<double[], double[]>();
        public double[][] Inputs => pairs.Keys.ToArray();
        public double[][] Outputs => pairs.Values.ToArray();
        public IEnumerable<KeyValuePair<double[], double[]>> Pairs => pairs;
        
        public int InputShape => Inputs.Max(x => x.Length);
        public int OutputShape => Outputs.Max(x => x.Length);

        public TrainingSetInputsTuple Input(params double[] inputs)
        {
            pairs.Add(inputs, new double[0]);
            return new TrainingSetInputsTuple {TrainingData = this, Inputs = inputs};
        }

        public class TrainingSetInputsTuple
        {
            public TrainingData TrainingData { get; set; } = new TrainingData();
            public double[] Inputs { get; set; } = new double[0];

            public TrainingData Output(params double[] outputs)
            {
                TrainingData.pairs[Inputs] = outputs;
                return TrainingData;
            }
        }
    }
}