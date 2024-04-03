package nn

import (
	"fmt"
	"github.com/chrissexton/goradient/value"
)

type Neuron struct {
	w      []*value.Value
	b      *value.Value
	nonLin bool
}

func NewNeuron(nin int, nonlin bool) *Neuron {
	return &Neuron{
		w:      make([]*value.Value, nin),
		b:      value.New(0, "b"),
		nonLin: nonlin,
	}
}

func (n *Neuron) Fwd(x []*value.Value) *value.Value {
	act := sum(n.w, x)
	if n.nonLin {
		return act.Tanh()
	}
	return act
}

func (n *Neuron) String() string {
	t := "Linear"
	if n.nonLin {
		t = "Tanh"
	}
	return fmt.Sprintf("%sNeuron(%d)", t, len(n.w))
}

func (n *Neuron) Parameters() []*value.Value {
	return append(n.w, n.b)
}

func sum(wi, xi []*value.Value) *value.Value {
	sum := value.New(0, "sum")
	for i := range wi {
		mul := wi[i].Mul(xi[i])
		sum = sum.Add(mul)
	}
	return sum
}
