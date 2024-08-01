package activation

import (
	"fmt"
	"testing"
)

func TestLinearReturnsOutputIsEqualToInput(t *testing.T) {
	for _, testCase := range []float64{
		0.6, -0.2, 0,
	} {
		t.Run(
			fmt.Sprintf("%f", testCase),
			func(t *testing.T) {
				if Linear(testCase) != testCase {
					t.Error("Non matching positive value detected")
				}
			},
		)
	}
}
