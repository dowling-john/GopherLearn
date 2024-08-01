package neural_network

import "GopherLearn/math"

type Neuron struct {
	Weights    []float64             `json:"weights"`
	Bias       float64               `json:"bias"`
	Activation func(float64) float64 `json:"-"`
}

func (neuron *Neuron) Forward(inputs []float64) float64 {
	return neuron.Activation(math.Sum(math.Dot(inputs, neuron.Weights)) + neuron.Bias)
}
