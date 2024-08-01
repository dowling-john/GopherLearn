package loss

import (
	"fmt"
	"testing"
)

type testCase struct {
	Actuals  []float64
	Forcasts []float64
	Mse      float64
}

func TestMeanSquaredError(t *testing.T) {

	testCases := []testCase{
		{
			[]float64{0.1, 0.547, 0.269},
			[]float64{0.1, 0.547, 0.269},
			0.0,
		},
		{
			[]float64{0.15, 0.57847, 0.99269},
			[]float64{0.1, 0.97, 0.00000069},
			0.141336148999892,
		},
	}

	for _, testCase := range testCases {
		t.Run(
			fmt.Sprintf("%f, %f", testCase.Forcasts, testCase.Actuals),
			func(t *testing.T) {
				fmt.Println(MeanSquaredError(testCase.Forcasts, testCase.Actuals))
				fmt.Println(testCase.Mse)
				if MeanSquaredError(testCase.Forcasts, testCase.Actuals) != testCase.Mse {
					t.Error(fmt.Sprintf("Expected meanSquaredError to return zero values %f", testCase.Mse))
				}
			},
		)
	}

}
