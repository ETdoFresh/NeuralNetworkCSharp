using System;

namespace NeuralNetwork
{
    public class Matrix
    {
        public static Random Random = new Random();

        public int rows;
        public int columns;
        public double[,] data;

        public double this[int row, int column]
        {
            get => data[row, column];
            set => data[row, column] = value;
        }

        public double this[int i]
        {
            get => data[i % rows, i / rows];
            set => data[i % rows, i / rows] = value;
        }

        public Matrix(int rows, int columns)
        {
            this.rows = rows;
            this.columns = columns;
            data = new double[rows, columns];
        }

        public Matrix(double[] data)
        {
            rows = data.Length;
            columns = 1;
            this.data = new double[rows, columns];
            for (var row = 0; row < rows; row++)
                this[row, 0] = data[row];
        }

        public Matrix(double[][] data)
        {
            rows = data.Length;
            columns = data[0].Length;
            this.data = new double[rows, columns];
            for (var row = 0; row < rows; row++)
            for (var column = 0; column < columns; column++)
                this[row, column] = data[row][column];
        }

        public Matrix(double[,] data)
        {
            rows = data.GetLength(0);
            columns = data.GetLength(1);
            this.data = new double[rows, columns];
            for (var row = 0; row < rows; row++)
            for (var column = 0; column < columns; column++)
                this[row, column] = data[row, column];
        }

        public Matrix(Matrix matrix)
        {
            rows = matrix.rows;
            columns = matrix.columns;
            data = new double[rows, columns];
            for (var row = 0; row < rows; row++)
            for (var column = 0; column < columns; column++)
                this[row, column] = matrix[row, column];
        }

        public Matrix Clone() => new Matrix(this);

        public Matrix Multiply(double scalar)
        {
            for (var i = 0; i < data.Length; i++)
                this[i] *= scalar;
            return this;
        }

        public Matrix Add(double scalar)
        {
            for (var row = 0; row < rows; row++)
            for (var column = 0; column < columns; column++)
                this[row, column] += scalar;
            return this;
        }
        
        public Matrix Subtract(double scalar)
        {
            for (var row = 0; row < rows; row++)
            for (var column = 0; column < columns; column++)
                this[row, column] -= scalar;
            return this;
        }

        public Matrix Randomize()
        {
            for (var row = 0; row < rows; row++)
            for (var column = 0; column < columns; column++)
                this[row, column] = Random.NextDouble() * 2 - 1;
            return this;
        }

        public Matrix Add(Matrix other)
        {
            if (!IsSameSize(this, other))
                throw new ArgumentException("Trying to add two matrices of different sizes");

            for (var row = 0; row < rows; row++)
            for (var column = 0; column < columns; column++)
                this[row, column] += other[row, column];
            return this;
        }
        
        public Matrix Subtract(Matrix other)
        {
            if (!IsSameSize(this, other))
                throw new ArgumentException("Trying to add two matrices of different sizes");

            for (var row = 0; row < rows; row++)
            for (var column = 0; column < columns; column++)
                data[row, column] -= other[row, column];
            return this;
        }

        public Matrix Multiply(Matrix other)
        {
            if (!IsSameSize(this, other))
                throw new ArgumentException("Trying to add two matrices of different sizes");

            for (var row = 0; row < rows; row++)
            for (var column = 0; column < columns; column++)
                data[row, column] *= other[row, column];
            return this;
        }

        public Matrix Map(Func<double, double> mapFunction)
        {
            for (var row = 0; row < rows; row++)
            for (var column = 0; column < columns; column++)
                data[row, column] = mapFunction(data[row, column]);
            return this;
        }

        public static Matrix Add(Matrix matrix, double scalar) => matrix.Clone().Add(scalar);
        public static Matrix Add(Matrix a, Matrix b) => a.Clone().Add(b);
        public static Matrix Subtract(Matrix matrix, double scalar) => matrix.Clone().Subtract(scalar);
        public static Matrix Subtract(Matrix a, Matrix b) => a.Clone().Subtract(b);
        public static Matrix Multiply(Matrix matrix, double scalar) => matrix.Clone().Multiply(scalar);
        public static Matrix Map(Matrix matrix, Func<double, double> mapFunction) => matrix.Clone().Map(mapFunction);
        
        public static Matrix Multiply(Matrix a, Matrix b)
        {
            if (a.columns != b.rows)
                throw new ArgumentException("Incompatible matrix sizes for Matrix Product");

            var result = new Matrix(a.rows, b.columns);
            for (var row = 0; row < result.rows; row++)
            for (var column = 0; column < result.columns; column++)
            for (var i = 0; i < a.columns; i++)
                result[row, column] += a[row, i] * b[i, column];
            return result;
        }

        public static Matrix Transpose(Matrix matrix)
        {
            var result = new Matrix(matrix.columns, matrix.rows);
            for (var row = 0; row < matrix.rows; row++)
            for (var column = 0; column < matrix.columns; column++)
                result[column, row] = matrix[row, column];
            return result;
        }

        public double[] ToArray()
        {
            var arr = new double[data.Length];
            for (var i = 0; i < arr.Length; i++)
                arr[i] = this[i];
            return arr;
        }
        
        public static bool IsSameSize(Matrix a, Matrix b)
        {
            return a.rows == b.rows && a.columns == b.columns;
        }

        public override string ToString()
        {
            var output = "Matrix\n";
            for (var row = 0; row < rows; row++)
            {
                for (var column = 0; column < columns; column++)
                    output += column == 0 ? $"{data[row, column]}" : $" {data[row, column]}";
                output += "\n";
            }

            return output;
        }
    }
}