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

func (n *NeuralNetwork) Backward(networkOutputs, expectedOutputs []float64) (errors [][]float64) {
	for l := len(n.Layers); l >= 0; l-- {
		if l == len(n.Layers) {
			errors = append(errors, n.Layers[l].Backward(expectedOutputs, networkOutputs, nil))
		}
		errors = append(errors, n.Layers[l].Backward(nil, nil, n.Layers[l+1]))
	}
	return
}
