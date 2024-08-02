Gopher Learn
============

This repo is purely for me to learn neural networks and how they work, if it helps you or you find it useful then brilliant


Initializing the Neural Network
-------------------------------

```go
    network := neural_network.NeuralNetwork{
        Layers: []*neural_network.FullyConnectedLayer{
            neural_network.NewFullyConnectedLayer(4, activation.ReLU),
            neural_network.NewFullyConnectedLayer(20, activation.ReLU),
            neural_network.NewFullyConnectedLayer(20, activation.ReLU),
            neural_network.NewFullyConnectedLayer(3, activation.ReLU),
        },
    }
```
