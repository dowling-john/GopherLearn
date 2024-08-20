package neuron

import (
	"GopherLearn/gopher_learn/activation"
	"GopherLearn/math"
	"math/rand"
)

type Neuron struct {
	Weights    []float64             `json:"weights"`
	Bias       float64               `json:"bias"`
	Activation activation.Activation `json:"-"`
	ErrorDelta float64               `json:"error_delta"`
	Inputs     []float64             `json:"inputs"`
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
	neuron.Inputs = inputs
	if neuron.Weights == nil {
		neuron.initializeWeights(len(inputs))
	}
	if neuron.Bias == 0.0 {
		neuron.initializeBias()
	}
	return neuron.Activation.Fire(math.Sum(math.Dot(inputs, neuron.Weights)) + neuron.Bias)
}

func (neuron *Neuron) UpdateWeight(index int, weight float64) {
	neuron.Weights[index] = weight
}

func (neuron *Neuron) UpdateBias(bias float64) {
	neuron.Bias = bias
}
