package layers

import "GopherLearn/gopher_learn/neuron"


type Layer interface {
	Forward(inputs[]float64) (NeuronOutputs []float64)
	GetNeurons() ([]*neuron.Neuron)
}