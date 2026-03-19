using LabMiniChatGPT_A7.DummyInterfaces;

namespace LabMiniChatGPT_A7.FakeMathOps;

public class FakeMathOps : IMathOps
{
    //All numbers are random, only for tests
    public float[] Softmax(ReadOnlySpan<float> logits)
    {
        float[] softArray = new float[logits.Length];

        for (int i = 0; i < softArray.Length; i++)
        {
            softArray[i] = 0.1f;
        }
        
        return softArray;
    }

    public float CrossEntropyLoss(ReadOnlySpan<float> logits, int target)
    {
        return 0.99f;
    }
}