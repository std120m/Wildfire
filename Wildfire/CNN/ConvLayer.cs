using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Wildfire.CNN
{
    class ConvLayer
    {
        public TensorSize InputSize { get; set; }
        public TensorSize OutputSize { get; set; }
        public List<Tensor> Filters { get; set; }
        public List<Tensor> Delta { get; set; }

        public List<double> Bias { get; set; }
        public List<double> DeltaBias { get; set; }

        public int Step { get; set; }
        public int Padding { get; set; }

        private int filtersCount;
        private int filterSize;
        static private Random random = new Random();

        public ConvLayer(TensorSize inputSize, int filtersCount, int filterSize, int step, int padding, double bias = 0)
        {
            InputSize = inputSize;
            Filters = new List<Tensor>();
            Delta = new List<Tensor>();
            Bias = new List<double>();
            DeltaBias = new List<double>();
            this.filtersCount = filtersCount;
            this.filterSize = filterSize;
            Step = step;
            Padding = padding;

            CalculateOutputSize();
            InitFilters();
            InitBias(bias);
            InitWeights();
        }

        public ConvLayer(int width, int height, int depth, int filtersCount, int filterSize, int step, int padding, double bias = 0)
            : this(new TensorSize(width, height, depth), filtersCount, filterSize, step, padding, bias) { }

        private void CalculateOutputSize()
        {
            OutputSize.Width = (InputSize.Width - filterSize + 2 * Padding) / Step + 1;
            OutputSize.Height = (InputSize.Height - filterSize + 2 * Padding) / Step + 1;
            OutputSize.Depth = filtersCount;
        }

        private void InitFilters()
        {
            for (int i = 0; i < filtersCount; i++)
            {
                Filters.Add(new Tensor(filterSize, filterSize, InputSize.Depth));
            }
        }
        private void InitBias(double value)
        {
            for (int i = 0; i < filtersCount; i++)
            {
                Bias.Add(value);
            }
        }

        private void InitWeights()
        {
            for (int filterId = 0; filterId < filtersCount; filterId++)
                for (int x = 0; x < filterSize; x++)
                    for (int y = 0; y < filterSize; y++)
                        for (int z = 0; z < InputSize.Depth; z++)
                            Filters[filterId].SetValue(x, y, z, random.NextDouble());
        }

        public Tensor Forward(Tensor input)
        {
            Tensor output = new Tensor(OutputSize);

            for (int filterId = 0; filterId < filtersCount; filterId++)
            {
                for (int y = 0; y < OutputSize.Height; y++)
                {
                    for (int x = 0; x < OutputSize.Width; x++)
                    {
                        double sum = Bias[filterId];

                        for (int i = 0; i < filterSize; i++)
                        {
                            for (int j = 0; j < filterSize; j++)
                            {
                                int y0 = Step * y + i - Padding;
                                int x0 = Step * x + j - Padding;

                                if (y0 < 0 || y0 >= InputSize.Height || x0 < 0 || x0 >= InputSize.Width)
                                    continue;

                                for (int z = 0; z < InputSize.Depth; z++)
                                    sum += input.GetValue(x0, y0, z) * Filters[filterId].GetValue(i, j, z);
                            }
                        }

                        output.SetValue(x, y, filterId, sum);
                    }
                }
            }

            return output;
        }

        public Tensor Backward(Tensor yTensor, Tensor xTensor)
        {
            TensorSize deltaSize = new TensorSize();
            deltaSize.Height = Step * (OutputSize.Height - 1) + 1;
            deltaSize.Width = Step * (OutputSize.Width - 1) + 1;
            deltaSize.Depth = OutputSize.Depth;

            Tensor delta = new Tensor(deltaSize);

            for (int z = 0; z < deltaSize.Depth; z++)
                for (int y = 0; y < deltaSize.Height; y++)
                    for (int x = 0; x < deltaSize.Width; x++)
                        delta.SetValue(x * Step, y * Step, z, yTensor.GetValue(x, y, z));

            for (int filterId = 0; filterId < filtersCount; filterId++)
            {
                for (int y = 0; y < OutputSize.Height; y++)
                {
                    for (int x = 0; x < OutputSize.Width; x++)
                    {
                        double d = delta.GetValue(x, y, filterId);

                        for (int i = 0; i < filterSize; i++)
                        {
                            for (int j = 0; j < filterSize; j++)
                            {
                                int y0 = y + i - Padding;
                                int x0 = x + j - Padding;


                                if (y0 < 0 || y0 >= InputSize.Height || x0 < 0 || x0 >= InputSize.Width)
                                    continue;

                                for (int z = 0; z < InputSize.Depth; z++)
                                {
                                    double temp = Delta[filterId].GetValue(j, i, z);
                                    temp += d * xTensor.GetValue(x0, y0, z);
                                    Delta[filterId].SetValue(j, i, z, temp);
                                }

                            }
                        }

                        DeltaBias[filterId] += d;
                    }
                }
            }

            int pad = filterSize - 1 - Padding;
            Tensor deltaX = new Tensor(InputSize);

            for (int y = 0; y < InputSize.Height; y++)
            {
                for (int x = 0; x < InputSize.Width; x++)
                {
                    for (int z = 0; z < InputSize.Depth; z++)
                    {
                        double sum = 0;

                        for (int i = 0; i < filterSize; i++)
                        {
                            for (int j = 0; j < filterSize; j++)
                            {
                                int y0 = y + i - pad;
                                int x0 = x + j - pad;

                                if (y0 < 0 || y0 >= deltaSize.Height || x0 < 0 || x0 >= deltaSize.Width)
                                    continue;

                                for (int filterId = 0; filterId < filtersCount; filterId++)
                                    sum += Filters[filterId].GetValue(filterSize - 1 - j, filterSize - 1 - i, z) * delta.GetValue(x0, y0, filterId);
                            }
                        }

                        deltaX.SetValue(x, y, z, sum);
                    }
                }
            }

            return deltaX;
        }

        public void UpdateWeights(double learningRate)
        {
            for (int filterId = 0; filterId < filtersCount; filterId++)
            {
                for (int y = 0; y < filterSize; y++)
                {
                    for (int x = 0; x < filterSize; x++)
                    {
                        for (int z = 0; z < InputSize.Depth; z++)
                        {
                            double temp = Filters[filterId].GetValue(x, y, z);
                            temp -= learningRate * Delta[filterId].GetValue(x, y, z);
                            Filters[filterId].SetValue(x, y, z, temp);
                            Delta[filterId].SetValue(x, y, z, 0);
                        }
                    }
                }

                Bias[filterId] -= learningRate * DeltaBias[filterId];
                DeltaBias[filterId] = 0;
            }
        }
    }
}
