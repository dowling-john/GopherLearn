package activation

type Relu struct{}

func (r *Relu) Fire(x float64) float64 {
	if x >= 0 {
		return x
	}
	return 0
}

func (r *Relu) GetDerivative(input float64) float64 {
	if input > 0 {
		return 1
	}
	return 0
}
