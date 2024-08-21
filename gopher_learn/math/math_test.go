package math

import (
	"fmt"
	"testing"
)

func TestSum(t *testing.T) {

	t.Run(
		"testing function 1", func(t *testing.T) {
			if Sum([]float64{4, 5}) != 9 {
				t.Error("Sum function failed")
			}
		},
	)

}

func TestDot(t *testing.T) {
	t.Run(
		"testing function 1", func(t *testing.T) {
			d := Dot([]float64{2, 3}, []float64{2, 3})
			fmt.Println(d)
			if d[0] != 4 && d[1] != 9 {
				t.Error("Dot function failed")
			}
		},
	)
}
