package activation

type Linear struct{}

func (l *Linear) Derivation(f float64) float64 {
	//TODO implement me
	panic("implement me")
}

func (l *Linear) Fire(x float64) float64 {
	return x
}
