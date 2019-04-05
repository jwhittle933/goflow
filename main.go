package main

import (
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func main() {
	root := op.NewScope()

	A := op.Placeholder(root.SubScope("input"), tf.Int64, op.PlaceholderShape(tf.MakeShape(2, 2)))
	x := op.Placeholder(root.SubScope("input"), tf.Int64, op.PlaceholderShape(tf.MakeShape(2, 1)))

	product := op.MatMul(root, A, x)

	graph, err := root.Finalize()
	if err != nil {
		panic(err.Error())
	}

	var sess *tf.Session
	sess, err = tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		panic(err.Error())
	}

	var matrix, column *tf.Tensor
	if matrix, err = tf.NewTensor([2][2]int64{{1, 2}, {-1, -2}}); err != nil {
		panic(err.Error())
	}

	if column, err = tf.NewTensor([2][1]int64{{10}, {100}}); err != nil {
		panic(err.Error())
	}

	var results []*tf.Tensor
	if results, err = sess.Run(map[tf.Output]*tf.Tensor{
		A: matrix,
		x: column,
	}, []tf.Output{product}, nil); err != nil {
		panic(err.Error())
	}

	for _, result := range results {
		fmt.Println("Result: ", result.Value().([][]int64))
	}

}
