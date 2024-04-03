package nn

import (
	"fmt"
	"testing"
)

func TestMLP(t *testing.T) {
	model := NewMLP(2, []int{16, 16, 1})
	fmt.Println(model)
	fmt.Println("Number of parameters: ", len(model.Parameters()))
}
