using System;
using System.Text;

namespace NeuralNetwork.MatrixModels
{
    public class Matrix
    {
        internal static Matrix Empty { get; } = new Matrix(0, 0);

        public int Rows { get; set; }
        public int Columns { get; set; }
        public double[] Values { get; set; }

        internal int Length => Rows * Columns;

        public double this[int i] { get => Values[i]; set => Values[i] = value; }

        public double this[int row, int column]
        {
            get => Values[row * Columns + column];
            set => Values[row * Columns + column] = value;
        }

        public Matrix() { }

        public Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;
            Values = new double[rows * columns];
        }

        public Matrix(double[] values)
        {
            Rows = values.Length;
            Columns = 1;
            Values = values;
        }

        public Matrix Transpose()
        {
            var originalValues = Clone();
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    this[j, i] = originalValues[i, j];
                }
            }
            return this;
        }

        public Matrix Add(Matrix matrix)
        {
            if (Rows != matrix.Rows || Columns != matrix.Columns)
            {
                throw new Exception("Matrices must have the same dimensions");
            }
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    this[i, j] += matrix[i, j];
                }
            }
            return this;
        }

        public Matrix Subtract(Matrix matrix)
        {
            if (Rows != matrix.Rows || Columns != matrix.Columns)
            {
                throw new Exception("Matrices must have the same dimensions");
            }
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    this[i, j] -= matrix[i, j];
                }
            }

            return this;
        }

        public Matrix Scale(Matrix matrix)
        {
            if (Rows != matrix.Rows || Columns != matrix.Columns)
                throw new Exception("Matrices must have the same dimensions");

            for (var row = 0; row < Rows; row++)
            for (var column = 0; column < Columns; column++)
                this[row, column] *= matrix[row, column];
            return this;
        }

        public Matrix Multiply(double scalar)
        {
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    this[i, j] *= scalar;
                }
            }
            return this;
        }

        public Matrix Divide(double scalar)
        {
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    this[i, j] /= scalar;
                }
            }
            return this;
        }

        public Matrix Inverse()
        {
            if (Rows != Columns)
            {
                throw new Exception("Matrix must be square");
            }

            for (var i = 0; i < Rows; i++)
            {
                this[i, i] = 1 / this[i, i];
                for (var j = 0; j < Rows; j++)
                {
                    if (i != j)
                    {
                        this[j, i] = -this[j, i] / this[i, i];
                    }
                }
            }
            return this;
        }

        public double Determinant()
        {
            if (Rows != Columns)
            {
                throw new Exception("Matrix must be square");
            }

            double determinant = 0;
            if (Rows == 1)
            {
                determinant = this[0, 0];
            }
            else if (Rows == 2)
            {
                determinant = this[0, 0] * this[1, 1] - this[0, 1] * this[1, 0];
            }
            else
            {
                for (var i = 0; i < Rows; i++)
                {
                    determinant += this[0, i] * Cofactor(0, i);
                }
            }

            return determinant;
        }

        private double Cofactor(int i, int i1)
        {
            double cofactor = 0;
            if (Rows == 2)
            {
                cofactor = this[0, 0] * this[1, 1] - this[0, 1] * this[1, 0];
            }
            else
            {
                var minor = new Matrix(Rows - 1, Columns - 1);
                for (var j = 0; j < Rows - 1; j++)
                {
                    for (var k = 0; k < Columns - 1; k++)
                    {
                        if (j < i && k < i1)
                        {
                            minor[j, k] = this[j + 1, k + 1];
                        }
                        else if (j < i && k >= i1)
                        {
                            minor[j, k] = this[j + 1, k];
                        }
                        else if (j >= i && k < i1)
                        {
                            minor[j, k] = this[j, k + 1];
                        }
                        else
                        {
                            minor[j, k] = this[j, k];
                        }
                    }
                }

                cofactor = minor.Determinant();
            }

            if ((i + i1) % 2 == 0)
            {
                cofactor = -cofactor;
            }

            return cofactor;
        }

        public void Randomize()
        {
            for (var i = 0; i < Rows; i++)
            for (var j = 0; j < Columns; j++)
                this[i, j] = Randomizer.Range(-1, 1);
        }

        public Matrix Map(Func<double, double> function)
        {
            for (var i = 0; i < Rows; i++)
            for (var j = 0; j < Columns; j++)
                this[i, j] = function(this[i, j]);
            return this;
        }

        public static Matrix Multiply(Matrix a, Matrix b)
        {
            if (a.Columns != b.Rows)
                throw new Exception("Matrix dimensions do not match");

            var result = new Matrix(a.Rows, b.Columns);
            for (var i = 0; i < a.Rows; i++)
            for (var j = 0; j < b.Columns; j++)
            for (var k = 0; k < a.Columns; k++)
                result[i, j] += a[i, k] * b[k, j];
            return result;
        }

        private Matrix Clone()
        {
            var clone = new Matrix(Rows, Columns);
            for (var i = 0; i < Values.Length; i++)
                clone.Values[i] = Values[i];
            return clone;
        }

        public double[] ToArray()
        {
            double[] array = new double[Rows * Columns];
            for (var i = 0; i < Rows; i++)
            for (var j = 0; j < Columns; j++)
                array[i * Columns + j] = this[i, j];
            return array;
        }

        public static Matrix Subtract(Matrix a, Matrix b)
        {
            var result = new Matrix(a.Rows, a.Columns);
            for (var i = 0; i < a.Rows; i++)
                result[i] = a[i] - b[i];
            return result;
        }

        public static Matrix Transpose(Matrix a)
        {
            var result = new Matrix(a.Columns, a.Rows);
            for (var i = 0; i < a.Rows; i++)
            for (var j = 0; j < a.Columns; j++)
                result[j, i] = a[i, j];
            return result;
        }

        public static Matrix Map(Matrix a, Func<double, double> func)
        {
            return a.Clone().Map(func);
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    sb.Append((j == 0 ? "" : ",") + this[i, j]);
                }

                if (Rows > 1 && i != Rows - 1) sb.Append("|");
            }

            return sb.ToString();
        }
    }
}