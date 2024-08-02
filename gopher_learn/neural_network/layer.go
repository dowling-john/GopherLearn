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

func (f *FullyConnectedLayer) Backward(expected, actuals []float64, previousLayer *FullyConnectedLayer) (errors []float64) {
	// This is the output Layer
	if previousLayer == nil {
		for i, neuron := range f.Neurons {
			errors = append(errors, actuals[i]-expected[i])
			neuron.Backward(nil, i, actuals[i], expected[i])
		}
		return
	}
	// Deal with hidden layers
	for i, neuron := range f.Neurons {
		errors = append(errors, neuron.Backward(previousLayer, i, 0.0, 0.0))
	}
	return
}

// ToDo: look at implementing a convolutional layer
