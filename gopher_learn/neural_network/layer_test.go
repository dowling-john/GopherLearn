package neural_network

import (
	"GopherLearn/gopher_learn/activation"
	"testing"
)

func TestInitializedLayerHasCorrectNumberOfNeurons(t *testing.T) {
	if len(NewFullyConnectedLayer(4, nil).Neurons) != 4 {
		t.Error("Layer should have 4 neurons")
	}
}

func TestInitializedLayerForwardReturnsTheCorrectNumberBasedOnNeurons(t *testing.T) {
	if len(NewFullyConnectedLayer(4, activation.Linear).Forward([]float64{0.1, .034, 0.66, 2.9})) != 4 {
		t.Error("Layer should have neuron outputs")
	}
}
