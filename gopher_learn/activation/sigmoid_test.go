package activation

import (
	"fmt"
	"testing"
)

func TestSigmoidActivationGivesCorrectOutput(t *testing.T) {
	testCases := [][]float64{
		{-0.6, 0.35434369377420455}, {0.6, 0.6456563062257954},
	}
	for _, testCase := range testCases {
		t.Run(
			fmt.Sprintf("%f, %f", testCase[0], testCase[1]),
			func(t *testing.T) {
				result := Sigmoid(testCase[0])
				if result != testCase[1] {
					t.Errorf(fmt.Sprintf("Incorrect values detected: %f expected: %f ", result, testCase[1]))
				}
			},
		)
	}
}
