using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Wildfire.CNN
{
    class TensorSize
    {
        public int Depth { get; set; }
        public int Height { get; set; }
        public int Width { get; set; }

        public TensorSize(int width = 0, int height = 0, int depth = 0)
        {
            Width = width;
            Height = height;
            Depth = depth;
        }
    }
}
