package neural_network

type (
	FullyConnectedLayer struct {
		Neurons []*Neuron `json:"neurons"`
	}
)

func (f *FullyConnectedLayer) Forward(inputs []float64) (NeuronOutputs []float64) {
	for _, neuron := range f.Neurons {
		NeuronOutputs = append(NeuronOutputs, neuron.Forward(inputs))
	}
	return NeuronOutputs
}

// ToDo: look at implementing a convolutional layer
