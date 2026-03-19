# Mini-ChatGPT: Stage 1 (Group A7)

> **Library:** `Lib.Models.TinyNN`  
> **Role:** Neural Language Model implementation using Embedding and Linear layers.

---

## Project Structure
According to the Stage 1 requirements, the library is strictly isolated and independently buildable:

| File | Responsibility |
| :--- | :--- |
| `TinyNNModel.cs` | Main orchestrator for forward pass and training steps. |
| `TinyNNWeights.cs` | State container for embeddings, weights, and biases. |
| `EmbeddingLayer.cs` | Logic for encoding context through embedding averaging. |
| `LinearHead.cs` | Logic for linear projection and backpropagation gradients. |
| `TinyNNModelFactory.cs` | Factory for creating new models or restoring from checkpoints. |

---

## Implementation Details

### Mathematical Operations
* **Forward Pass**:
    * `EncodeContext`: Averages embeddings of context tokens to form a hidden state.
    * `Project`: Maps hidden states to vocabulary logits.
* **Backward Pass (`TrainStep`)**:
    * Calculates Cross-Entropy Loss and propagates gradients to update model parameters.
    * Implements `weight -= lr * gradient` logic for SGD.

### Essential Features
* **Context Slicing**: Automatically slices input to the last `ContextSize` tokens to maintain consistency.
* **Contract Fingerprint**: Includes a stable `GetContractFingerprint()` for system-wide compatibility checks.

---

##  How to Run
Use the standard .NET 8.0 CLI to manage the project:

```
# Build the library
dotnet build

# Execute Unit and Integration tests
dotnet test

# Generate code coverage report
dotnet test --collect:"XPlat Code Coverage"