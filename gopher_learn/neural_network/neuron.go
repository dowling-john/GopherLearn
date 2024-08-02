package neural_network

import (
	"GopherLearn/math"
	"math/rand"
)

type Neuron struct {
	Weights    []float64             `json:"weights"`
	Bias       float64               `json:"bias"`
	Activation func(float64) float64 `json:"-"`
}

func (neuron *Neuron) initializeWeights(input_size int) {
	for i := 0; i < input_size; i++ {
		neuron.Weights = append(neuron.Weights, rand.Float64())
	}
}

func (neuron *Neuron) initializeBias() {
	neuron.Bias = rand.Float64()
}

func (neuron *Neuron) Forward(inputs []float64) float64 {
	if neuron.Weights == nil {
		neuron.initializeWeights(len(inputs))
	}
	if neuron.Bias == 0.0 {
		neuron.initializeBias()
	}
	return neuron.Activation(math.Sum(math.Dot(inputs, neuron.Weights)) + neuron.Bias)
}
