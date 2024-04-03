package nn

import (
	"fmt"
	"github.com/chrissexton/gograd/value"
	"strings"
)

type Layer struct {
	neurons []*Neuron
}

func NewLayer(nin, nout int, nonLin bool) *Layer {
	neurons := make([]*Neuron, nout)
	for i, _ := range neurons {
		neurons[i] = NewNeuron(nin, nonLin)
	}
	return &Layer{
		neurons: neurons,
	}
}

func (l *Layer) Fwd(x []*value.Value) []*value.Value {
	out := []*value.Value{}
	for _, n := range l.neurons {
		out = append(out, n.Fwd(x))
	}
	return out
}

func (l *Layer) String() string {
	neurons := []string{}
	for _, n := range l.neurons {
		neurons = append(neurons, n.String())
	}
	return fmt.Sprintf("Layer of [%s]", strings.Join(neurons, ","))
}

func (l *Layer) Parameters() []*value.Value {
	out := []*value.Value{}
	for _, n := range l.neurons {
		out = append(out, n.Parameters()...)
	}
	return out
}
