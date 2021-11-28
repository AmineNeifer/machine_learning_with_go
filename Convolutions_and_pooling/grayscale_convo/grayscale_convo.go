package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/sbinet/npyio/npz"
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
	var mul float32
	output := tensor.New(tensor.WithShape(a1.Shape()[0], a1.Shape()[1]), tensor.Of(tensor.Float32))
	for i := 0; i < a1.Shape()[0]; i++ {
		for j := 0; j < a1.Shape()[1]; j++ {
			value_1, _ := a1.At(i, j)
			value_2, _ := a2.At(i, j)
			// fmt.Printf("value_1 = %v   value_2 = %v\n", value_1, value_2)
			val1 := InterfaceToFloat32(value_1)
			val2 := InterfaceToFloat32(value_2)
			// fmt.Printf("val1 = %v   val2 = %v\n", val1, val2)
			mul = val1 * val2
			output.SetAt(mul, i, j)
		}
	}
	return output
}

const (
	num_img  = 50000
	input_h  = 28                     // input height
	input_w  = 28                     // input width
	filter_h = 3                      // filter height
	filter_w = 3                      // filter width
	output_h = input_h - filter_h + 1 // output height
	output_w = input_w - filter_w + 1 // output width
)

// Convert interface to integer
func InterfaceToInt(i interface{}) int {
	s := fmt.Sprintf("%v", i)
	sum, _ := strconv.Atoi(s)
	return sum
}

// convert interface to uint8
func InterfaceToUint8(i interface{}) uint8 {
	s := fmt.Sprintf("%v", i)
	sum, _ := strconv.Atoi(s)

	return uint8(sum)
}

// convert interface to uint8
func InterfaceToFloat32(i interface{}) float32 {
	s := fmt.Sprintf("%v", i)
	f, _ := strconv.ParseFloat(s, 32)
	return float32(f)
}

// Convert Dense to integer
func DenseToInt(d *tensor.Dense) int {
	s := fmt.Sprintf("%v", d)
	sum, _ := strconv.Atoi(s)
	return sum
}

// Convert Dense to integer
func DenseToFloat(d *tensor.Dense) float32 {
	s := fmt.Sprintf("%v", d)
	f, _ := strconv.ParseFloat(s, 32)
	return float32(f)
}

// Convert Slice to Image
func TensorViewToImage(slice tensor.View) image.Image {
	height, width := slice.Shape()[0], slice.Shape()[1]
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for py := 0; py < height; py++ {
		for px := 0; px < width; px++ {
			// Image point (px, py) represents complex value z.
			val, _ := slice.At(py, px)
			val_float32 := InterfaceToFloat32(val)
			val_float32 *= 255
			img.Set(px, py, color.Gray{uint8(val_float32)})
		}
	}
	return img
}

// Write image to file png
func WriteImage(filename string, img image.Image) {
	out, err := os.Create(filename)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer out.Close()

	if err = png.Encode(out, img); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

// Pick picture, n < 50000
func PickImage(dense *tensor.Dense, n int) tensor.View {
	slice, err := dense.Slice(makeRS(n, n+1), makeRS(0, dense.Shape()[0]), makeRS(0, dense.Shape()[1]))
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	return slice
}

func main() {
	f, err := os.Open("data/MNIST.npz")
	if err != nil {
		log.Fatalf("could not open npz file: %+v", err)
	}
	defer f.Close()

	var X_train []float32
	err = npz.Read(f, "X_train.npy", &X_train)
	if err != nil {
		log.Fatalf("could not read value from npz file: %+v", err)
	}
	// reshape X_train to (num_img, height, width) [50000 28 28]
	input := tensor.New(tensor.WithShape(50000, 28, 28), tensor.WithBacking(X_train))
	// choose the picture, I picked 6th number (index 5)
	img_pickd := PickImage(input, 0)
	fmt.Printf("%#v", img_pickd)
	// convert slice into a grayscale image
	img := TensorViewToImage(img_pickd)
	// save the imgByte to file
	WriteImage("./output.png", img)
	// convolve image with our filter
	filter := tensor.New(tensor.WithShape(filter_h, filter_w), tensor.WithBacking([]float32{1, 0, -1, 1, 0, -1, 1, 0, -1}))

	output := tensor.New(tensor.WithShape(output_h, output_w), tensor.Of(tensor.Float32))
	for w := 0; w < output_w; w++ {
		for h := 0; h < output_h; h++ {
			slice, _ := img_pickd.Slice(makeRS(h, h+filter_h), makeRS(w, w+filter_w))
			v := slice.(*tensor.Dense)
			mul := Multiply(v, filter)
			// fmt.Printf("%v ",mul)
			s, _ := mul.Sum()
			output.SetAt(DenseToFloat(s), h, w)
		}
	}
	img1 := TensorViewToImage(output)
	WriteImage("./convolved.png", img1)
}
