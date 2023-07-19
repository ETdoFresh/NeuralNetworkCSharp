using System;
using System.Linq;

namespace NeuralNetwork.MatrixModels
{
    public class SequentialModel : Model
    {
        public override Matrix Predict(Matrix inputs)
        {
            var layer = inputs;
            outputAs.Clear();
            outputAs.Add(inputs);
            for (var i = 0; i < Weights.Count; i++)
            {
                layer = Matrix.Multiply(Weights[i], layer);
                layer.Add(Biases[i]);
                layer.Map(Activation.Sigmoid);
                outputAs.Add(layer);
            }

            return layer;
        }

        private void BackPropagate(Matrix expectedOutput)
        {
            var outputErrors = Matrix.Subtract(expectedOutput, outputAs.Last());
            for (var i = outputAs.Count - 1; i > 0; i--)
            {
                var outputs = outputAs[i];
                if (i < outputAs.Count - 1)
                {
                    var weightT = Matrix.Transpose(Weights[i]);
                    outputErrors = Matrix.Multiply(weightT, outputErrors);
                }

                var gradients = Matrix.Map(outputs, Activation.SigmoidDerivative);
                gradients.Scale(outputErrors);
                gradients.Multiply(LearningRate);
                var previousOutputsT = Matrix.Transpose(outputAs[i - 1]);
                var weightDeltas = Matrix.Multiply(gradients, previousOutputsT);
                Weights[i - 1].Add(weightDeltas);
                Biases[i - 1].Add(gradients);
            }
        }

        public override void Train(int epochs, int epochDisplayInterval)
        {
            var targetEpochs = TrainedEpochs + epochs;
            DisplayOnConsole(1);
            while (TrainedEpochs < targetEpochs)
            {
                DisplayOnConsole(epochDisplayInterval);
                foreach (var kvp in TrainingData.Pairs)
                {
                    var input = kvp.Key;
                    var expectedOutput = kvp.Value;
                    Predict(input);
                    BackPropagate(expectedOutput);
                }

                TrainedEpochs++;
            }
        }

        private void DisplayOnConsole(int epochDisplayInterval)
        {
            var displayedEpoch = TrainedEpochs + 1;
            var errorSum = 0.0;
            foreach (var kvp in TrainingData.Pairs)
            {
                var input = kvp.Key;
                var expectedOutput = kvp.Value;
                var output = Predict(input);
                errorSum += ComputeDisplayError(expectedOutput, output);
                if (displayedEpoch % epochDisplayInterval == 0)
                {
                    var inputs = string.Join(", ", input.Values.Select(i => i.ToString("0.00")).ToArray());
                    var expectedOutputs = string.Join(", ", expectedOutput.Values.Select(i => i.ToString("0.00")).ToArray());
                    var outputs = string.Join(", ", output.Values.Select(i => i.ToString("0.00")).ToArray());
                    Console.WriteLine(
                        $"Epoch: {displayedEpoch} Input: {inputs} Expected: {expectedOutputs} Output: {outputs}");
                }
            }

            if (displayedEpoch % epochDisplayInterval == 0)
                Console.WriteLine("Epoch " + displayedEpoch + " error: " + errorSum);
        }
        
        private double ComputeDisplayError(Matrix expectedOutput, Matrix output)
        {
            var error = 0.0;
            for (var i = 0; i < expectedOutput.Length; i++)
            {
                error += Math.Abs(expectedOutput[i] - output[i]);
            }

            return error;
        }

        public override void PrintInitialError()
        {
            var error = 0.0;
            foreach (var kvp in TrainingData.Pairs)
            {
                var input = kvp.Key;
                var expectedOutput = kvp.Value;
                var output = Predict(input);
                error += ComputeDisplayError(expectedOutput, output);
            }

            error /= TrainingData.Pairs.Count();
            Console.WriteLine("Initial error: " + error);
        }
    }
}