package activation

type Activation interface {
	Fire(float64) float64
	GetDerivative(float64) float64
}
