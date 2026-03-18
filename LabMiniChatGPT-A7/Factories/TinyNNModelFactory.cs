using Configuration.LabMiniChatGPT_A7;
using System.Text.Json;
using Lib.MathCore; 
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;

namespace LabMiniChatGPT_A7.Factories;

public class TinyNNModelFactory
{
    public TinyNNModel CreateNew(TinyNNConfig config, IMathOps mathOps)
    {
        TinyNNWeights weights = new TinyNNWeights(config);
        
        return new TinyNNModel(config, weights, mathOps);
    }

    public TinyNNModel FromPayload(JsonElement payload, int vocabSize, IMathOps mathOps)
    {
        var configParams = payload.GetProperty("config");
        var config = new TinyNNConfig(vocabSize, configParams.GetProperty("embedingSize").GetInt32(), configParams.GetProperty("contextSize").GetInt32());
        
        var weightParams = payload.GetProperty("weights");
        var weights = JsonSerializer.Deserialize<TinyNNWeights>(weightParams.GetRawText());

        return TinyNNModel(config, weights, mathOps);
    }
}