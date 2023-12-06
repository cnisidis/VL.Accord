using System;

using Accord.Statistics.Models.Markov.Learning;
using Accord.Statistics.Distributions.Univariate;
using Accord.Statistics.Models.Markov;
using Accord.Statistics.Distributions.Multivariate;
using Accord.Statistics.Models.Markov.Topology;
using Accord.Statistics.Distributions.Fitting;

namespace VL.Accord.HMM
{

    //http://accord-framework.net/docs/html/T_Accord_Statistics_Models_Fields_HiddenConditionalRandomField_1.htm
    public class HMM
    {
        public HiddenMarkovClassifierLearning<Independent<NormalDistribution>, double[]> teacher;
        public HiddenMarkovClassifier<Independent<NormalDistribution>, double[]> hmm;
        public double logLikelihood;
        public HMM(int dimensions=4, double tolerance=0.001, int iterations=100, int topology=5, double regularization= 1e-5)
        {
            teacher = new HiddenMarkovClassifierLearning<Independent<NormalDistribution>, double[]>()
            {

                Learner = (i) => new BaumWelchLearning<Independent<NormalDistribution>, double[]>()
                {
                    Topology = new Forward(topology), // this value can be found by trial-and-error
                    Emissions = (s) => new Independent<NormalDistribution>(dimensions: dimensions),
                    Tolerance = tolerance,
                    MaxIterations = iterations,
                   


                    FittingOptions = new IndependentOptions()
                    {
                        InnerOption = new NormalOptions() { Regularization = regularization }
                    }
                }
            };

        }

        public void Learn(double[][][] data, int[] labels)
        {
            //teacher.ParallelOptions.MaxDegreeOfParallelism = 1;
            this.hmm = teacher.Learn(data, labels);
            this.logLikelihood = teacher.LogLikelihood;
        }

        public object Decide(double [][] input)
        {
            var output = this.hmm.Decide(input);
            return output;
        }


    }
}
