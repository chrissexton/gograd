package value

import (
	"testing"
)

func TestExampleNetwork1(t *testing.T) {
	x1 := New(2.0, "x1")
	x2 := New(0.0, "x2")
	w1 := New(-3.0, "w1")
	w2 := New(1.0, "w2")
	b := New(6.8813735870195432, "b")
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
	t.Log(o.Print())
	if !(o.data > 0.7 && o.data < 7.1) {
		t.Fatal(o.data)
	}
}

func TestExampleNetwork2(t *testing.T) {
	x1 := New(2.0, "x1")
	x2 := New(0.0, "x2")
	w1 := New(-3.0, "w1")
	w2 := New(1.0, "w2")
	b := New(6.8813735870195432, "b")
	x1w1 := x1.Mul(w1)
	x1w1.SetName("x1*w1")
	x2w2 := x2.Mul(w2)
	x2w2.SetName("x2*w2")
	x1w1x2w2 := x1w1.Add(x2w2)
	x1w1x2w2.SetName("x1*w1 + x2*w2")
	n := x1w1x2w2.Add(b)
	n.SetName("n")
	n2 := n.Mul(New(2, "2"))
	n2.SetName("MulI2")
	e := n2.Exp()
	num := e.Sub(New(1, "o"))
	num.SetName("num")
	den := e.AddI(1)
	den.SetName("den")
	o := num.Div(den)
	o.SetName("o")
	o.Backward()
	t.Log(o.Print())
}

func TestReusedNode(t *testing.T) {
	a := New(3.0, "a")
	b := a.Add(a)
	b.SetName("b")
	b.Backward()
	t.Log(b.Print())
}

func TestReusedNode2(t *testing.T) {
	a := New(-2.0, "a")
	b := New(3.0, "b")
	d := a.Mul(b)
	d.SetName("d")
	e := a.Add(b)
	e.SetName("e")
	f := d.Mul(e)
	f.SetName("f")
	f.Backward()
	t.Log(f.Print())
}
