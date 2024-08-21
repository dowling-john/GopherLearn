package optimizers

import (
	"GopherLearn/gopher_learn/loss"
	"GopherLearn/gopher_learn/neural_network"
	"GopherLearn/gopher_learn/math"
	"fmt"
	//"github.com/kr/pretty"
)

type SGD struct {
	Epochs       int
	BatchSize    int
	LearningRate float64
	LossFunction loss.Loss
}

// Optimize the neural network
func (s *SGD) Optimize(network *neural_network.NeuralNetwork, optimisationData [][]float64, optimizationLabels [][]float64) (err error) {
	fmt.Println("Optimising the Network")
	for epoch := 0; epoch < s.Epochs; epoch++ {
		fmt.Printf("Epoch %d\n", epoch)
		s.processBatches(optimisationData, optimizationLabels, network)
	}
	fmt.Println("Finished Optimising the Network")
	return
}

// Forward pass should return a matrix like follows
//
//                         [I1, I2]
//         [HiddenLayerOutput1,  HiddenLayerOutput2]                        <- This layer had 2 neurons in it
// [HiddenLayerOutput1, HiddenLayerOutput2, HiddenLayerOutput3]             <- This layer had 3 neurons in it
//                     [HiddenLayerOutput1]                                 <- This layer had 1 neurons and is the output layer
func (s *SGD) forwardPass(inputValues []float64, network *neural_network.NeuralNetwork) (outputMatrix [][]float64) {
	nextLayerInputs := inputValues
	for _, layer := range network.Layers {
		layerOutputs := layer.Forward(nextLayerInputs)
		outputMatrix = append(outputMatrix, layerOutputs)
		nextLayerInputs = layerOutputs
	}
	return outputMatrix
}

func (s *SGD) processBatches(inputValues, inputTargets [][]float64, network *neural_network.NeuralNetwork) {
	errorDeltas := [][]float64{}
	errorAverages := []float64{}
	outputMatrix := [][][]float64{}

	//Get all the output values for the otimization data
	for _, b := range inputValues {
		outputMatrix = append(outputMatrix, s.forwardPass(b, network))
	}

	// Get loss for each neruron in the output layer for each example in the optimization data
	for _, r := range outputMatrix {
		outputErrorDeltas := []float64{}
		for i, j := range r[len(r)-1:] {
			outputErrorDeltas = append(outputErrorDeltas, s.LossFunction.GetLoss(j, inputTargets[i]))
		}
		errorDeltas = append(errorDeltas, outputErrorDeltas)
	}

	// Average the error of each neuron error across the training batch
	for n, _ := range network.GetOuputLayer().GetNeurons() {
		d := []float64{}
		for _, j := range errorDeltas {
			d = append(d, j[n])
		}
		errorAverages = append(errorAverages, math.Avg(d))
	}

	//Use the error values to optimize the network

	//loop through layers in reverse order
	for layer := len(network.Layers)-1; layer >= 0; layer-- {
		fmt.Printf("Layer: %v", layer)
		// Update the neurons induvidually 
		for neuronIndex, neuron := range network.Layers[layer].GetNeurons() {
			if layer == len(network.Layers)-1 {
				// Get the derivative of the activation function
				neuronActivationDerivative := neuron.Activation.GetDerivative(errorAverages[neuronIndex])
				for j, weight := range neuron.Weights {
					gradientForweight := neuronActivationDerivative * outputMatrix[0][layer-1][j]
					fmt.Printf("old weight -> : %v", neuron.Weights[j])
					neuron.Weights[j] = weight - s.LearningRate * gradientForweight
					fmt.Printf("new weight -> : %v", neuron.Weights[j])

				}
				// Update the bias
				gradientForBias := neuronActivationDerivative * outputMatrix[0][layer-1][neuronIndex]
				fmt.Printf("old bias -> : %v", neuron.Bias)
				neuron.Bias = neuron.Bias - s.LearningRate * gradientForBias
				fmt.Printf("new bias -> : %v", neuron.Bias)
				continue
			}
		}
	
	
	}

}
