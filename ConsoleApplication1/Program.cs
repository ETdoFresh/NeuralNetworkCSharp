using NeuralNetwork;

namespace ConsoleApplication1
{
    // Example Neural Network for XOR Learning

    // Neurons
    // Input Layer  Hidden Layer(s)  Output Layer
    // Input1       Hidden1          Output1
    // Input2       Hidden2

    // Connections
    // Input1-Hidden1, Input1-Hidden2, Input2-Hidden1, Input2-Hidden2
    // Hidden1-Output1, Hidden2-Output1

    public static class Program
    {
        public static void Main(string[] args)
        {
            SetupXorProblem(out var neuralNetwork, out var expectedResults);
            //SetupEchoProblem(out var neuralNetwork, out var expectedResults);
            neuralNetwork.PrintInitialError(expectedResults);
            neuralNetwork.Train(expectedResults, 50000, 1000);
        }

        private static void SetupXorProblem(out NetworkUsingMatrices neuralNetwork, out Batch expectedResults)
        {
            neuralNetwork = new NetworkUsingMatrices()
                .CreateInputLayerNeurons(2)
                .CreateHiddenLayerNeurons(4)
                //.CreateHiddenLayerNeurons(4)
                .CreateOutputLayerNeurons(1)
                .SetLearningRate(0.1);

            expectedResults = new Batch()
                .Input(0, 0).Output(0)
                .Input(0, 1).Output(1)
                .Input(1, 0).Output(1)
                .Input(1, 1).Output(0);
        }

        private static void SetupEchoProblem(out NetworkUsingMatrices neuralNetwork, out Batch expectedResults)
        {
            neuralNetwork = new NetworkUsingMatrices()
                .CreateInputLayerNeurons(1)
                .CreateHiddenLayerNeurons(1)
                .CreateOutputLayerNeurons(1);
            expectedResults = new Batch().Input(0).Output(0).Input(1).Output(1);
        }
    }
}