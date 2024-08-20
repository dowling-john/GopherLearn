package optimizers

import "GopherLearn/gopher_learn/neural_network"

type Optimizer interface {
	Optimize(network *neural_network.NeuralNetwork, optimisationData [][]float64, optimizationLabels []float64) error
}
