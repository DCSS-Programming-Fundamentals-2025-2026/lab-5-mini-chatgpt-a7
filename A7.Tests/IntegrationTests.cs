using LabMiniChatGPT_A7;
using LabMiniChatGPT_A7.Configuration;
using LabMiniChatGPT_A7.FakeMathOps;
using LabMiniChatGPT_A7.Layers;
using LabMiniChatGPT_A7.State;

namespace A7.Tests;

[TestFixture]
public class IntegrationTests
{
    [Test]
    public void Integration_TrainStep_ExecutesPipelineAndUpdatesWeights()
    {
        var config = new TinyNNConfig (10,4,3);
        var weights = new TinyNNWeights(config);
        var fakeMath = new FakeMathOps();
        var model = new TinyNNModel(config, weights, fakeMath);
        
        int[] context = new[] {1, 2, 5};
        int target = 7; //<=8
        float lr = 0.1f;
        float initialWeight = weights.OutputWeights[0][0];

        var loss = model.TrainStep(context, target, lr); //=0.99f

        Assert.Multiple(() =>
        {
            Assert.That(loss, Is.EqualTo(0.99f));
            Assert.That(weights.OutputWeights[0][0], Is.Not.EqualTo(initialWeight));
        });
    }
}