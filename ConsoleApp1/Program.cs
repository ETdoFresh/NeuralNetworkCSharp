using NeuralNetwork.MatrixModels;

namespace ConsoleApp1
{
    public static class Program
    {
        private const double LearningRate = 0.1;
        private const int Epochs = 50000;
        private const int EpochDisplayInterval = 1000;

        public static void Main(string[] args)
        {
            // Create a new neural network
            var model = AndModel;
            model.PrintInitialError();
            model.Train(Epochs, EpochDisplayInterval);
            model.WriteToJson();

            // Read existing network and train more
            // var model = Model.ReadFromJson();
            // model.SetLearningRate(LearningRate);
            // model.SetTrainingData(AndTrainingData);
            // model.Train(Epochs, EpochDisplayInterval);
            // model.WriteToJson();
            // Console.WriteLine(model.Predict(new Matrix(new double[] {0, 0})));
            // Console.WriteLine(model.Predict(new Matrix(new double[] {0, 1})));
            // Console.WriteLine(model.Predict(new Matrix(new double[] {1, 0})));
            // Console.WriteLine(model.Predict(new Matrix(new double[] {1, 1})));
        }

        private static Model EchoModel => new SequentialModel()
            .SetLearningRate(LearningRate)
            .SetTrainingData(EchoTrainingData)
            .AddInputLayer()
            .AddLayer(1)
            .AddOutputLayer();

        private static TrainingData EchoTrainingData => new TrainingData()
            .Input(0).Output(0)
            .Input(1).Output(1);

        private static Model XOrModel => new SequentialModel()
            .SetLearningRate(LearningRate)
            .SetTrainingData(XOrTrainingData)
            .AddInputLayer()
            .AddLayer(4)
            .AddLayer(4)
            .AddOutputLayer();

        private static TrainingData XOrTrainingData => new TrainingData()
            .Input(0, 0).Output(0)
            .Input(0, 1).Output(1)
            .Input(1, 0).Output(1)
            .Input(1, 1).Output(0);

        private static Model AndModel => new SequentialModel()
            .SetLearningRate(LearningRate)
            .SetTrainingData(AndTrainingData)
            .AddInputLayer()
            .AddLayer(4)
            .AddLayer(4)
            .AddOutputLayer();

        private static TrainingData AndTrainingData => new TrainingData()
            .Input(0, 0).Output(0)
            .Input(0, 1).Output(0)
            .Input(1, 0).Output(0)
            .Input(1, 1).Output(1);
    }
}