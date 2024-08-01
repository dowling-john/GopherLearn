package gopher_learn

import "math"

func BinaryStep(x float64) float64 {
	if x >= 0 {
		return 1
	}
	return 0
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(x*(-1)))
}
