using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Wildfire.CNN
{
    class Tensor
    {
        public TensorSize Size { get; set; }
        //private List<double> values;
        private double[,,] values;

        public Tensor(TensorSize size)
        {
            Size = size;
            values = new double[size.Width, size.Height, size.Depth];
            InitValue();
        }

        public Tensor(int width, int height, int depth) : this(new TensorSize(width, height, depth)) { }

        private void InitValue()
        {
            for (int z = 0; z < Size.Depth; z++)
            {
                for (int y = 0; y < Size.Height; y++)
                {
                    for (int x = 0; x < Size.Width; x++)
                    {
                        values[x, y, z] = 0;
                    }
                }
            }
        }

        public double GetValue(int x, int y, int z)
        {
            //return values[z * Width * Height + y * Depth + x];
            return values[x, y, z];
        }

        public void SetValue(int x, int y, int z, double value)
        {
            values[x, y, z] = value;
        }

        public string GetValues()
        {
            string result = string.Empty;

            for (int z = 0; z < Size.Depth; z++)
            {
                for (int y = 0; y < Size.Height; y++)
                {
                    for (int x = 0; x < Size.Width; x++)
                    {
                        result += GetValue(x, y, z).ToString() + " ";
                    }
                    result += "\n";
                }
            }

            return result;
        }
    }
}
