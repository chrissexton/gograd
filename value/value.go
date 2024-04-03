package value

import (
	"fmt"
	"log"
	"math"
	"slices"
	"strings"
)

type Value struct {
	name     string
	data     float64
	grad     float64
	backward func()
	prev     []*Value
	op       string
}

func New(data float64, label string) *Value {
	v := Value{}
	v.name = label
	v.data = data
	v.backward = func() {}
	v.prev = []*Value{}
	return &v
}

func (v *Value) String() string {
	return fmt.Sprintf("%s: %f, %f", v.name, v.data, v.grad)
}

func (v *Value) SetName(name string) {
	v.name = name
}

func (v *Value) setPrev(children ...*Value) {
	v.prev = append(v.prev, children...)
}

func (v *Value) setOp(op string) {
	v.op = op
}

func (v *Value) setBackward(f func()) {
	v.backward = f
}

func (v *Value) AddF(other float64) *Value {
	return v.Add(New(other, ""))
}

func (v *Value) AddI(other int) *Value {
	return v.Add(New(float64(other), ""))
}

func (v *Value) Add(other *Value) *Value {
	out := New(v.data+other.data, "+")
	out.setOp("+")
	out.setBackward(func() {
		v.grad += 1 * out.grad
		other.grad += 1 * out.grad
	})
	out.setPrev(v, other)
	return out
}

func (v *Value) Neg() *Value {
	return v.MulI(-1)
}

func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Neg())
}

func (v *Value) MulF(other float64) *Value {
	return v.Mul(New(other, ""))
}

func (v *Value) MulI(other int) *Value {
	return v.Mul(New(float64(other), ""))
}

func (v *Value) Mul(other *Value) *Value {
	out := New(v.data*other.data, "*")
	out.setOp("*")
	out.setBackward(func() {
		v.grad += other.data * out.grad
		other.grad += v.data * out.grad
	})
	out.setPrev(v, other)
	return out
}

func (v *Value) Div(other *Value) *Value {
	return v.Mul(other.Pow(New(-1, "**-1")))
}

func (v *Value) Pow(other *Value) *Value {
	out := New(math.Pow(v.data, other.data), "^")
	out.setOp(fmt.Sprintf("^%.2f", other.data))
	out.setBackward(func() {
		v.grad += other.data * math.Pow(v.data, other.data-1) * out.grad
	})
	out.setPrev(v)
	return out
}

func (v *Value) Exp() *Value {
	out := New(math.Exp(v.data), "exp")
	out.setOp("exp")
	out.setBackward(func() {
		log.Printf("expSetBackward: %.2f += %.2f * %.2f = %.2f", v.grad, out.data, out.grad, v.grad+out.data*out.grad)
		v.grad = out.data * out.grad
	})
	out.setPrev(v)
	return out
}

func (v *Value) Tanh() *Value {
	n := v.data
	t := (math.Exp(2*n) - 1) / (math.Exp(2*n) + 1)
	out := New(t, "tanh")
	out.setOp("tanh")
	out.setBackward(func() {
		v.grad = (1 - math.Pow(t, 2)) * out.grad
	})
	out.setPrev(v)
	return out
}

func (v *Value) Relu() *Value {
	newValue := v.data
	if v.data < 0 {
		newValue = 0
	}
	out := New(newValue, "ReLU")
	out.setOp("ReLU")
	out.setBackward(func() {
		value := out.data * 0.5
		if out.data <= 0 {
			value = 0
		}
		v.grad += value
	})
	out.setPrev(v)
	return out
}

func (v *Value) topo() []*Value {
	topo := []*Value{}
	visited := map[*Value]bool{}
	var build_topo func(*Value)
	build_topo = func(vv *Value) {
		if !visited[vv] {
			visited[vv] = true
			for _, child := range vv.prev {
				build_topo(child)
			}
		}
		topo = append(topo, vv)
	}
	build_topo(v)
	return topo
}

func (v *Value) Print() string {
	out := ""
	for _, vv := range v.topo() {
		out += "\n" + vv.String()
	}
	return strings.TrimSpace(out)
}

func (v *Value) Backward() {
	topo := v.topo()
	v.grad = 1
	slices.Reverse(topo)
	for _, vv := range topo {
		vv.backward()
	}
}
