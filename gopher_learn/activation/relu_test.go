package activation

import (
	"fmt"
	"testing"
)

func TestReluActivationGivesZeroOnNegativeValue(t *testing.T) {
	testCases := []float64{
		-0.6,
	}

	for _, testCase := range testCases {
		t.Run(
			fmt.Sprintf("%f", testCase),
			func(t *testing.T) {
				if ReLU(testCase) > 0 {
					t.Error("Negative values detected")
				}
			})
	}
}

func TestReluActivationGivesCorrectValueWithPositiveValue(t *testing.T) {
	testCases := []float64{
		0.6,
	}

	for _, testCase := range testCases {
		t.Run(
			fmt.Sprintf("%f", testCase),
			func(t *testing.T) {
				if ReLU(testCase) != testCase {
					t.Error("Non matching positive value detected")
				}
			})
	}
}
