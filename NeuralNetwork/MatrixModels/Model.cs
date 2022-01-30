using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace NeuralNetwork.MatrixModels
{
    public abstract class Model
    {
        public string ModelType { get; set; }
        public double LearningRate { get; set; } = 0.1;
        public string LossFunction { get; set; } = "Undefined/Mean Squared Error";
        public int TrainedEpochs { get; set; }
        public List<Matrix> Weights { get; set; } = new();
        public List<Matrix> Biases { get; set; } = new();
        protected TrainingData TrainingData { get; private set; } = new TrainingData();
        protected List<Matrix> outputAs { get; set; } = new();
        private int _inputNeuronCount = -1;


        public Model()
        {
            ModelType = GetType().Name;
        }

        public Model AddLayer(int numberOfNeurons)
        {
            if (_inputNeuronCount == -1)
            {
                _inputNeuronCount = TrainingData.InputShape;
            }
            else
            {
                Weights.Add(Weights.Count == 0
                    ? new Matrix(numberOfNeurons, _inputNeuronCount)
                    : new Matrix(numberOfNeurons, Weights.Last().Rows));
                Biases.Add(new Matrix(numberOfNeurons, 1));
            }
            return this;
        }

        public Model AddLayers(params int[] neuronCounts)
        {
            foreach (var neuronCount in neuronCounts)
            {
                AddLayer(neuronCount);
            }

            return this;
        }

        public Model AddInputLayer() => AddLayer(TrainingData.InputShape);

        public Model AddOutputLayer()
        {
            AddLayer(TrainingData.OutputShape);
            Weights.ForEach(w => w.Randomize());
            Biases.ForEach(b => b.Randomize());
            return this;
        }

        public abstract void Train(int epochs, int epochDisplayInterval);
        public abstract Matrix Predict(Matrix input);
        public abstract void PrintInitialError();

        public Model SetLearningRate(double learningRate)
        {
            LearningRate = learningRate;
            return this;
        }

        public Model SetTrainingData(TrainingData trainingData)
        {
            TrainingData = trainingData;
            return this;
        }

        public void WriteToJson()
        {
            var json = JsonSerializer.Serialize(this);
            File.WriteAllText("model.json", json);
        }

        public static Model? ReadFromJson()
        {
            var json = File.ReadAllText("model.json");
            return JsonSerializer.Deserialize<Model>(json,
                new JsonSerializerOptions {Converters = {new JsonModelConverter()}});
        }
    }
}