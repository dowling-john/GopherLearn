package neural_network

import "GopherLearn/gopher_learn/layers"

type NeuralNetwork struct {
	Layers       []layers.Layer         			`json:"layers"`
	LossFunction func(float64, float64) float64 	`json:"loss_function"`
}

func (n *NeuralNetwork) Forward(inputs []float64) []float64 {
	for _, layer := range n.Layers {
		inputs = layer.Forward(inputs)
	}
	return inputs
}

func (n * NeuralNetwork) GetOuputLayer() layers.Layer {
	return n.Layers[len(n.Layers)-1]
}