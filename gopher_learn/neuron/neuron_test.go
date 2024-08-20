package neuron

import (
	"GopherLearn/gopher_learn/activation"
	"testing"
)

func TestNeuronsInitializeWeightsIfNotAlreadySet(t *testing.T) {
	neuron := Neuron{
		Weights: nil, Bias: 0.04, Activation: activation.ReLU,
	}
	neuron.Forward([]float64{0.2, 0.4, 0.69})
}

func TestNeuronsInitializeBiasesIfNotAlreadySet(t *testing.T) {
	neuron := Neuron{
		Weights: []float64{.02, 4.5, 0.69}, Bias: 0.0, Activation: activation.ReLU,
	}
	neuron.Forward([]float64{0.2, 0.4, 0.69})
}

func TestNeuronsWeightsDoNotInitializeIfPreSet(t *testing.T) {
	x := []float64{.02, 4.5, 0.69}
	neuron := Neuron{
		Weights: x, Bias: 0.0, Activation: activation.ReLU,
	}
	neuron.Forward([]float64{0.2, 0.4, 0.69})
	for i, weight := range neuron.Weights {
		if weight != x[i] {
			t.Error("Weights have changed from init values")
		}
	}
}

func TestNeuronsBiasDoNotInitializeIfPreSet(t *testing.T) {
	x := 4.2
	neuron := Neuron{
		Weights: []float64{.02, 4.5, 0.69}, Bias: x, Activation: activation.ReLU,
	}
	neuron.Forward([]float64{0.2, 0.4, 0.69})

	if neuron.Bias != x {
		t.Error("Bias has changed from init value")
	}

}
