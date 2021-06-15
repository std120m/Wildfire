using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Wildfire.CNN
{
    [Serializable]
    class Synapse
    {
        public Neuron Input { get; set; }
        public Neuron Output { get; set; }
        public double Weight { get; set; }
        static Random random = new Random();
        public Synapse(Neuron input, Neuron output)
        {
            Input = input;
            Output = output;
            Weight = random.NextDouble();
        }

        public Synapse(Neuron input, Neuron output, double weight) : this(input, output)
        {
            Weight = weight;
        }
    }
}
