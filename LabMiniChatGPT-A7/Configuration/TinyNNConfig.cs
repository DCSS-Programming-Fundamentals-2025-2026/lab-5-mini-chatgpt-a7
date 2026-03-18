namespace Configuration.LabMiniChatGPT_A7;

public record TinyNNConfig()
{
    public int VocabSize { get; init; }
    public int EmbeddingSize { get; init; } = 32;
    public int ContextSize { get; init; } = 8;
}