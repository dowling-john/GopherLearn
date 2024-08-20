package layers


type Layer interface {
	Forward(inputs[]float64) (NeuronOutputs []float64)
}