package layers

import (
	"GopherLearn/gopher_learn/activation"
	"GopherLearn/gopher_learn/neuron"
)

type (
	FullyConnectedLayer struct {
		Neurons []*neuron.Neuron `json:"neurons"`
	}
)

func NewFullyConnectedLayer(number_of_neurons int, activation activation.Activation) *FullyConnectedLayer {
	var neurons []*neuron.Neuron
	for i := 0; i < number_of_neurons; i++ {
		neurons = append(neurons, &neuron.Neuron{Activation: activation})
	}
	return &FullyConnectedLayer{Neurons: neurons}
}

func (f *FullyConnectedLayer) Forward(inputs []float64) (NeuronOutputs []float64) {
	for _, neuron := range f.Neurons {
		NeuronOutputs = append(NeuronOutputs, neuron.Forward(inputs))
	}
	return NeuronOutputs
}

func (f *FullyConnectedLayer) GetNeurons() []*neuron.Neuron {
	return f.Neurons
}