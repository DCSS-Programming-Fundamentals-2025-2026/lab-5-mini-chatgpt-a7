using System.ComponentModel.DataAnnotations;
using Configuration.LabMiniChatGPT_A7;

namespace LabMiniChatGPT_A7;

public class TinyNNWeights
{
    public float[][] Embeddings { get; set; }
    public float[][] OutputWeights { get; set; }
    public float[] OutputBias { get; set; }

    private readonly float minRange = -0.1f;
    private readonly float maxRange = 0.1f;

    public TinyNNWeights(TinyNNConfig config)
    {
        Embeddings = new float[config.VocabSize][];
        for (int i = 0; i < config.VocabSize; i++)
        {
            Embeddings[i] = new float[config.EmbeddingSize];
            for (int j = 0; j < config.EmbeddingSize; j++)
            {
                Random temp = new Random();
                Embeddings[i][j] = (float)temp.NextDouble() * (maxRange - minRange) + minRange;
            }
        }
        
        OutputWeights = new float[config.EmbeddingSize][];
        for (int i = 0; i < config.EmbeddingSize; i++)
        {
            OutputWeights[i] = new float[config.VocabSize];
            for (int j = 0; j < config.VocabSize; j++)
            {
                Random temp = new Random();
                OutputWeights[i][j] = (float)temp.NextDouble() * (maxRange - minRange) + minRange;
            }
        }
        
        OutputBias = new float[config.VocabSize];
        for (int i = 0; i < config.VocabSize; i++)
        {
            Random temp = new Random();
            OutputBias[i] = (float)temp.NextDouble() * (maxRange - minRange) + minRange;
        }
    }
}