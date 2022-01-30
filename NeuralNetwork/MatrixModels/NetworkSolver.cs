using System;
using System.Linq;

namespace NeuralNetwork.MatrixModels
{
    public class NetworkSolver
    {
        private int Epochs { get; set; } = 100;
        private double LearningRate { get; set; } = 0.1;
        private NeuralNetwork NeuralNetwork { get; } = new();
        private TrainingData TrainingData { get; set; } = new();

        public NetworkSolver AddTrainingSet(TrainingData trainingData)
        {
            TrainingData = trainingData;
            return this;
        }

        public NetworkSolver CreateInputLayerFromTrainingSet()
        {
            if (TrainingData == null) throw new Exception("Training set is not set");
            if (TrainingData.Inputs == null) throw new Exception("Training set has no inputs");
            if (TrainingData.Inputs.Length == 0) throw new Exception("Training set has no inputs");
            if (TrainingData.Inputs.Any(x => x.Length == 0)) throw new Exception("Training set has empty inputs");
            NeuralNetwork.AddLayer(TrainingData.Inputs[0].Length);
            return this;
        }

        public NetworkSolver CreateHiddenLayer(int size)
        {
            NeuralNetwork.AddLayer(size);
            return this;
        }

        public NetworkSolver CreateOutputLayerFromTrainingSet()
        {
            if (TrainingData == null) throw new Exception("Training set is not set");
            if (TrainingData.Outputs == null) throw new Exception("Training set has no outputs");
            if (TrainingData.Outputs.Length == 0) throw new Exception("Training set has no outputs");
            if (TrainingData.Outputs.Any(x => x.Length == 0)) throw new Exception("Training set has empty outputs");
            NeuralNetwork.AddLayer(TrainingData.Outputs[0].Length);
            return this;
        }

        // public void PrintInitialNetworkError()
        // {
        //     var error = 0.0;
        //     foreach (var (input, expectedOutput) in TrainingData.Pairs)
        //     {
        //         var output = NeuralNetwork.Compute(input);
        //         error += ComputeDisplayError(expectedOutput, output);
        //     }
        //
        //     error /= TrainingData.Pairs.Count();
        //     Console.WriteLine("Initial error: " + error);
        // }

        public NetworkSolver SetEpochs(int epochs)
        {
            Epochs = epochs;
            return this;
        }

        public NetworkSolver SetLearningRate(double learningRate)
        {
            LearningRate = learningRate;
            return this;
        }

        // public void TrainNetwork(int epochDisplayInterval = 100)
        // {
        //     var epoch = 0;
        //     while (epoch <= Epochs)
        //     {
        //         var averageError = 0.0;
        //         foreach (var (input, expectedOutput) in TrainingData.Pairs)
        //         {
        //             var output = NeuralNetwork.Compute(input);
        //             NeuralNetwork.BackPropagate(expectedOutput, LearningRate);
        //             averageError += ComputeDisplayError(expectedOutput, output);
        //             if (epoch % epochDisplayInterval == 0) Console.WriteLine("Epoch: " + epoch + " Input: " + string.Join(", ", input) + " Expected output: " + string.Join(", ", expectedOutput) + " Output: " + string.Join(", ", output));
        //         }
        //
        //         averageError /= TrainingData.Pairs.Count();
        //         if (epoch % epochDisplayInterval == 0) Console.WriteLine("Epoch " + epoch + " average error: " + averageError);
        //         epoch++;
        //     }
        // }

        private double ComputeDisplayError(Matrix expectedOutput, Matrix output)
        {
            var error = 0.0;
            for (var i = 0; i < expectedOutput.Rows; i++)
            {
                for (var j = 0; j < expectedOutput.Columns; j++)
                {
                    error += Math.Abs(expectedOutput[i, j] - output[i, j]);
                }
            }
            return error;
        }
    }
}