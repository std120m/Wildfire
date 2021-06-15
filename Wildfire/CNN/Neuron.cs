using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Wildfire.CNN
{
    [Serializable]
    class Neuron
    {
        public double Value { get; set; }
        private double a;
        Random random = new Random();

        public Neuron()
        {
            a = 0.5;
            Value = random.NextDouble();
        }

        public Neuron(double value) : this()
        {
            Value = value;
        }

        public void Activation(double input)
        {
            Value = 1.0 / (1 + Math.Exp(-a * input));
        }
    }
}
