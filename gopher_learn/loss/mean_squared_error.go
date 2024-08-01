package loss

import (
	m "GopherLearn/math"
	"math"
)

func MeanSquaredError(forcast []float64, actual []float64) float64 {
	var x []float64
	for i, e := range forcast {
		x = append(x, actual[i]-e)
	}
	return float64(1) / float64(len(forcast)) * math.Pow(m.Sum(x), 2)
}
