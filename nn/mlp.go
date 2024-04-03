package nn

import (
	"fmt"
	"github.com/chrissexton/goradient/value"
	"strings"
)

type MLP struct {
	layers []*Layer
}

func NewMLP(nin int, nouts []int) *MLP {
	sizes := []int{nin}
	sizes = append(sizes, nouts...)
	mlp := MLP{layers: []*Layer{}}
	for i := 0; i < len(nouts); i++ {
		nonLin := i != len(nouts)-1
		mlp.layers = append(mlp.layers, NewLayer(sizes[i], sizes[i+1], nonLin))
	}
	return &mlp
}

func (m *MLP) Fwd(x []*value.Value) []*value.Value {
	for _, layer := range m.layers {
		x = layer.Fwd(x)
	}
	return x
}

func (m *MLP) Parameters() []*value.Value {
	out := []*value.Value{}
	for _, layer := range m.layers {
		out = append(out, layer.Parameters()...)
	}
	return out
}

func (m *MLP) String() string {
	layers := []string{}
	for _, l := range m.layers {
		layers = append(layers, l.String())
	}
	return fmt.Sprintf("MLP of [%s]", strings.Join(layers, ","))
}
