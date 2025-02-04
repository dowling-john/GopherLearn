package math

func Dot(x, y []float64) (z []float64) {
	for i, _ := range x {
		z = append(z, x[i]*y[i])
	}
	return
}

func Sum(x []float64) (sum float64) {
	for _, l := range x {
		sum += l
	}
	return
}
