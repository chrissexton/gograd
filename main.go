package main

import (
	"fmt"
	"github.com/chrissexton/gograd/value"
)

func main() {
	x1 := value.New(2.0, "x1")
	x2 := value.New(0.0, "x2")
	w1 := value.New(-3.0, "w1")
	w2 := value.New(1.0, "w2")
	b := value.New(6.8813735870195432, "b")
	x1w1 := x1.Mul(w1)
	x1w1.SetName("x1*w1")
	x2w2 := x2.Mul(w2)
	x2w2.SetName("x2*w2")
	x1w1x2w2 := x1w1.Add(x2w2)
	x1w1x2w2.SetName("x1*w1 + x2*w2")
	n := x1w1x2w2.Add(b)
	n.SetName("n")
	o := n.Tanh()
	o.SetName("o")
	o.Backward()
	fmt.Println(o.Print())
}
