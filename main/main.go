package main

import (
	"GopherLearn/gopher_learn/activation"
	"GopherLearn/gopher_learn/neural_network"
	"encoding/json"
	"fmt"
)

func main() {

	neuron := &neural_network.Neuron{
		Weights: []float64{-0.2, 0.2}, Bias: 0.01, Activation: activation.ReLU,
	}

	neuron2 := &neural_network.Neuron{
		Weights: []float64{0.2, -0.6}, Bias: 0.451, Activation: activation.ReLU,
	}

	neuron3 := &neural_network.Neuron{
		Weights: []float64{0.2, 0.6, 0.5, 0.1}, Bias: 0.451, Activation: activation.Linear,
	}

	layer := &neural_network.FullyConnectedLayer{
		Neurons: []*neural_network.Neuron{
			neuron, neuron2,
		},
	}

	layer2 := &neural_network.FullyConnectedLayer{
		Neurons: []*neural_network.Neuron{
			neuron, neuron, neuron2, neuron2,
		},
	}

	layer3 := &neural_network.FullyConnectedLayer{
		Neurons: []*neural_network.Neuron{
			neuron3,
		},
	}

	network := &neural_network.NeuralNetwork{
		Layers: []*neural_network.FullyConnectedLayer{
			layer, layer2, layer3,
		},
	}
	_, _ = json.MarshalIndent(network, "", "  ")
	n := network.Forward([]float64{0.4, 0.3})
	fmt.Println(n)
}
