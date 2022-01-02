using System.Collections;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public abstract class ArrayClass<T> : IEnumerable<T>
    {
        private readonly T[] _values;

        protected ArrayClass(params T[] values) => _values = values;
        
        public T this[int index] => _values[index];
        public int Length => _values.Length;
        
        public IEnumerator<T> GetEnumerator() => ((IEnumerable<T>)_values).GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => _values.GetEnumerator();
        
        public override string ToString()
        {
            var output = "";
            for (var i = 0; i < _values.Length; i++)
                output += i == 0 ? $"{_values[i]}" : $" {_values[i]}";
            return output;
        }

        public static implicit operator T[](ArrayClass<T> arrayClass) => arrayClass._values;
    }
}