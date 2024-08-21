package loss

import (
	"GopherLearn/gopher_learn/math"
	math2 "math"
)

type MeanSquaredError struct{}

func (m *MeanSquaredError) GetDerivative(input, target float64) float64 {
	return 2 * (input - target)
}

func (m *MeanSquaredError) GetLoss(forecast []float64, actual []float64) float64 {
	var x []float64
	for i, e := range forecast {
		x = append(x, actual[i]-e)
	}
	return float64(1) / float64(len(forecast)) * math2.Pow(math.Sum(x), 2)
}
