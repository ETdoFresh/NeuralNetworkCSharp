using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.ArrayModels
{
    /// <summary>
    /// Array classes not quite working yet
    /// </summary>
    public class TrainingData
    {
        private Dictionary<double[], double[]> pairs = new();
        public double[][] Inputs => pairs.Keys.ToArray();
        public double[][] Outputs => pairs.Values.ToArray();
        public IEnumerable<KeyValuePair<double[], double[]>> Pairs => pairs;
        
        public int InputShape => Inputs.Max(x => x.Length);
        public int OutputShape => Outputs.Max(x => x.Length);

        public TrainingSetInputsTuple Input(params double[] inputs)
        {
            pairs.Add(inputs, Array.Empty<double>());
            return new TrainingSetInputsTuple {TrainingData = this, Inputs = inputs};
        }

        public class TrainingSetInputsTuple
        {
            public TrainingData TrainingData { get; init; } = new();
            public double[] Inputs { get; init; } = Array.Empty<double>();

            public TrainingData Output(params double[] outputs)
            {
                TrainingData.pairs[Inputs] = outputs;
                return TrainingData;
            }
        }
    }
}