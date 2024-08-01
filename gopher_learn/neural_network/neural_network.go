package neural_network

type NeuralNetwork struct {
	Layers []*FullyConnectedLayer `json:"layers"`
}

func (n *NeuralNetwork) Forward(inputs []float64) []float64 {
	for _, layer := range n.Layers {
		inputs = layer.Forward(inputs)
	}
	return inputs
}
