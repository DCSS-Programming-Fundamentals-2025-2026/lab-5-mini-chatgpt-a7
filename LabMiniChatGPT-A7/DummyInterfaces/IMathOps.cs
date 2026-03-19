namespace LabMiniChatGPT_A7.DummyInterfaces;

public interface IMathOps
{
    float[] Softmax(ReadOnlySpan<float> logits);
    float CrossEntropyLoss(ReadOnlySpan<float> logits, int target);
}