package main

import (
	"GopherLearn/gopher_learn/activation"
	"fmt"
)

func main() {
	//network := neural_network.NeuralNetwork{
	//	Layers: []*neural_network.FullyConnectedLayer{
	//		neural_network.NewFullyConnectedLayer(4, activation.ReLU),
	//		neural_network.NewFullyConnectedLayer(20, activation.ReLU),
	//		neural_network.NewFullyConnectedLayer(20, activation.ReLU),
	//		neural_network.NewFullyConnectedLayer(3, activation.ReLU),
	//	},
	//}
	//fmt.Println(network.Forward([]float64{0.05, 0.2, 2.5, 0.1}))

	result := activation.Sigmoid(-0.6)
	fmt.Println(result)
}
