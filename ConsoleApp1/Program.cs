using NeuralNetwork;

namespace ConsoleApp1
{
    public static class Program
    {
        private const double LearningRate = 0.1;
        private const int Epochs = 50000;
        private const int EpochDisplayInterval = 1000;

        public static void Main(string[] args)
        {
            RandomUtil.SetSeed(42);
            Connection.SetRandomWeightRange(-1.00, 1.00);
            Neuron.SetRandomBiasRange(-1.00, 1.00);
            
            // Create a new neural network
            var model = AndModel;
            var trainingData = AndTrainingData;
            
            model.PrintInitialError(trainingData);
            model.Train(Epochs, EpochDisplayInterval, trainingData);
            //model.WriteToJson();

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

        private static Model EchoModel => new Model()
            .SetLearningRate(LearningRate)
            .AddInputLayer(EchoTrainingData.InputShape)
            .AddFullyConnectedLayer(1)
            .AddFullyConnectedOutputLayer(EchoTrainingData.OutputShape);

        private static TrainingData EchoTrainingData => new TrainingData()
            .Input(0).Output(0)
            .Input(1).Output(1);

        private static Model XOrModel => new Model()
            .SetLearningRate(LearningRate)
            .AddInputLayer(XOrTrainingData.InputShape)
            .AddFullyConnectedLayer(4)
            .AddFullyConnectedLayer(4)
            .AddFullyConnectedOutputLayer(XOrTrainingData.OutputShape);

        private static TrainingData XOrTrainingData => new TrainingData()
            .Input(0, 0).Output(0)
            .Input(0, 1).Output(1)
            .Input(1, 0).Output(1)
            .Input(1, 1).Output(0);

        private static Model AndModel => new Model()
            .SetLearningRate(LearningRate)
            .AddInputLayer(AndTrainingData.InputShape)
            .AddFullyConnectedLayer(4)
            .AddFullyConnectedLayer(4)
            .AddFullyConnectedOutputLayer(AndTrainingData.OutputShape);

        private static TrainingData AndTrainingData => new TrainingData()
            .Input(0, 0).Output(0)
            .Input(0, 1).Output(0)
            .Input(1, 0).Output(0)
            .Input(1, 1).Output(1);
    }
}