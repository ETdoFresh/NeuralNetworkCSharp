using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Batch : List<Pair>
    {
        private static readonly Random random = new Random();

        public Batch(params Pair[] pairs) : this((IEnumerable<Pair>)pairs) { }

        public Batch(IEnumerable<Pair> pairs) => AddRange(pairs);

        public Batch Clone()
        {
            return new Batch(this);
        }

        public Batch Randomize()
        {
            var clone = Clone();
            for (var i = Count - 1; i > 0; i--)
            {
                var k = random.Next(i + 1);
                var swapValue = this[k];
                clone[k] = clone[i];
                clone[i] = swapValue;
            }

            return clone;
        }
    }
}