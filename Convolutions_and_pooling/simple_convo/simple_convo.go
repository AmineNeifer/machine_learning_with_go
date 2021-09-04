package main

import (
	"fmt"
	"math/rand"
	"strconv"
	"time"

	"gorgonia.org/tensor"
)

type rs struct {
	start, end, step int
}

func (s rs) Start() int { return s.start }
func (s rs) End() int   { return s.end }
func (s rs) Step() int  { return s.step }

// makeRS creates a ranged slice. It takes an optional step param.
func makeRS(start, end int, opts ...int) rs {
	step := 1
	if len(opts) > 0 {
		step = opts[0]
	}
	return rs{
		start: start,
		end:   end,
		step:  step,
	}
}
// RandIntArray creates a slice with random elements in [0, n)
func RandIntArray(n int, low int, high int) []int {
	var s []int
	rand.Seed(time.Now().Unix())
	for i := 0; i < n; i++ {
		s = append(s, rand.Intn(high-low)+low)
	}
	return s
}

// element-wise multiplication
func Multiply(a1, a2 *tensor.Dense) *tensor.Dense {
	var mul int;
	output := tensor.New(tensor.WithShape(a1.Shape()[0], a1.Shape()[1]), tensor.Of(tensor.Int))
	for i:= 0; i < a1.Shape()[0]; i++ {
		for j:= 0; j < a1.Shape()[1]; j++ {
			value_1, _ := a1.At(i, j)
			value_2, _ := a2.At(i, j)

			val1 := InterfaceToInt(value_1)
			val2 := InterfaceToInt(value_2)
			mul = val1 * val2 
			output.SetAt(mul, i, j);
		}
	}
	return output
}

const (
	input_h  = 8 // input height
	input_w  = 8 // input width
	filter_h = 3 // filter height
	filter_w = 3 // filter width
	output_h = input_h - filter_h + 1 // output height
	output_w = input_w - filter_w + 1 // output width
)

// Convert interface to integer
func InterfaceToInt(i interface{}) int {
	s := fmt.Sprintf("%v", i)
	sum, _ := strconv.Atoi(s)
	return sum
}

// Convert Dense to integer
func DenseToInt(d *tensor.Dense) int {
	s := fmt.Sprintf("%v", d)
	sum, _ := strconv.Atoi(s)
	return sum
}

func main() {
	input := tensor.New(tensor.WithShape(input_h, input_w), tensor.WithBacking(RandIntArray(64, 0, 10)))
	fmt.Printf("\ninput:\n%v", input)
	filter := tensor.New(tensor.WithShape(filter_h, filter_w), tensor.WithBacking([]int{1, 0, -1, 1, 0, -1, 1, 0, -1}))

	output := tensor.New(tensor.WithShape(output_h, output_w), tensor.Of(tensor.Int))
	for w := 0; w < output_w; w++ {
		for h := 0; h < output_h; h++ {
			slice, _ := input.Slice(makeRS(h, h+filter_h), makeRS(w, w+filter_w))
			v := slice.(*tensor.Dense)
			mul := Multiply(v, filter)
			s, _ := mul.Sum()
			output.SetAt(DenseToInt(s), h, w)
		}
	}
	fmt.Printf("\nOutput:\n%v", output)
}

