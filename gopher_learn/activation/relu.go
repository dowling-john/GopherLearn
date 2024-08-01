package activation

// ReLU implementation of the following function
// f(x) = max(0, x)
func ReLU(x float64) float64 {
	if x >= 0 {
		return x
	}
	return 0
}
