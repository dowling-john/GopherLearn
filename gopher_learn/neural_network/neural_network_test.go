package neural_network

import (
	"GopherLearn/gopher_learn/activation"
	"testing"
)

func TestNetworkGivesCorrectNumberOfOutputsForOutputLayer(t *testing.T) {
	network := &NeuralNetwork{
		Layers: []*FullyConnectedLayer{
			NewFullyConnectedLayer(4, activation.Linear),
			NewFullyConnectedLayer(2, activation.Linear),
		},
	}
	if len(network.Forward([]float64{0.2, -0.1, 2.66, 4.26})) != 2 {
		t.Error("Network sending incorrect number of outputs")
	}
}
