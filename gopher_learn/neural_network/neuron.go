package neural_network

import (
	"GopherLearn/math"
	"math/rand"
)

type Neuron struct {
	Weights    []float64             `json:"weights"`
	Bias       float64               `json:"bias"`
	Activation func(float64) float64 `json:"-"`
	ErrorDelta float64               `json:"error_delta"`
}

func (neuron *Neuron) initializeWeights(input_size int) {
	for i := 0; i < input_size; i++ {
		neuron.Weights = append(neuron.Weights, rand.Float64())
	}
}

func transferDerivative(output float64) float64 {
	return output * (1.0 - output)
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

// This function needs a little work, once you go past the first hidden layer I am not sure where the error delta comes from as this is only calculated for the output layer
func (neuron *Neuron) Backward(previousLayer *FullyConnectedLayer, currentNeuronIndex int, output, error float64) (e float64) {
	// if this neuron is in the output layer
	if previousLayer == nil {
		neuron.ErrorDelta = error * transferDerivative(output)
		return
	}
	for _, previousNeuron := range previousLayer.Neurons {
		e += (previousNeuron.Weights[currentNeuronIndex] * previousNeuron.ErrorDelta)
	}
	return
}
