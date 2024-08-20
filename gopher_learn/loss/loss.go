package loss

type Loss interface {
	GetLoss(forecast []float64, actual []float64) float64
}
