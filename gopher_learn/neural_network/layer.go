package neural_network

type (
	FullyConnectedLayer struct {
		Neurons []*Neuron `json:"neurons"`
	}
)

func NewFullyConnectedLayer(number_of_neurons int, activation_func func(float64) float64) *FullyConnectedLayer {
	var neurons []*Neuron
	for i := 0; i < number_of_neurons; i++ {
		neurons = append(neurons, &Neuron{Activation: activation_func})
	}
	return &FullyConnectedLayer{Neurons: neurons}
}

func (f *FullyConnectedLayer) Forward(inputs []float64) (NeuronOutputs []float64) {
	for _, neuron := range f.Neurons {
		NeuronOutputs = append(NeuronOutputs, neuron.Forward(inputs))
	}
	return NeuronOutputs
}

// ToDo: look at implementing a convolutional layer
