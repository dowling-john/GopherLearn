package main

import (
	"GopherLearn/gopher_learn/activation"
	"GopherLearn/gopher_learn/layers"
	"GopherLearn/gopher_learn/loss"
	"GopherLearn/gopher_learn/neural_network"
	"GopherLearn/gopher_learn/optimizers"
	"fmt"
)

var (
	xorProblem = [][]float64{
		{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0},
	}

	xorActuals = [][]float64{
		{0.0}, {1.0}, {1.0}, {0.0},
	}
)


func main() {
	network := &neural_network.NeuralNetwork{
		Layers: []layers.Layer{
			layers.NewFullyConnectedLayer(2, &activation.Relu{}),
			layers.NewFullyConnectedLayer(2, &activation.Relu{}),
			layers.NewFullyConnectedLayer(1, &activation.Relu{}),
		},
	}

	o := optimizers.SGD{
		LearningRate: 0.001,
		Epochs:       2,
		BatchSize:    4,
		LossFunction: &loss.MeanSquaredError{},
	}

	if err := o.Optimize(network, xorProblem, xorActuals); err != nil {
		fmt.Println(err)
	}
}
