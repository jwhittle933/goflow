## GoFlow

Combining TensorFlow and Golang for ML and NLP

### Example 
```go
package main

import (
	"fmt"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func main() {
	// Let's describe what we want: create the graph

	// We want to define two placeholder to fill at runtime
	// the first placeholder A will be a [2, 2] tensor of integers
	// the second placeholder x will be a [2, 1] tensor of intergers

	// Then we want to compute Y = Ax

	// Create the first node of the graph: an empty node, the root of our graph
	root := op.NewScope()

	// Define the 2 placeholders
	A := op.Placeholder(root.SubScope("input"), tf.Int64, op.PlaceholderShape(tf.MakeShape(2, 2)))
  x := op.Placeholder(root.SubScope("input"), tf.Int64, op.PlaceholderShape(tf.MakeShape(2, 1)))
  fmt.Println(A.Op.Name(), x.Op.Name())

	// Define the operation node that accepts A & x as inputs
	product := op.MatMul(root, A, x)

	// Every time we passed a `Scope` to an operation, we placed that
	// operation **under** that scope.
	// As you can see, we have an empty scope (created with NewScope): the empty scope
	// is the root of our graph and thus we denote it with "/".

	// Now we ask tensorflow to build the graph from our definition.
	// The concrete graph is created from the "abstract" graph we defined
	// using the combination of scope and op.

	graph, err := root.Finalize()
	if err != nil {
		// It's useless trying to handle this error in any way:
		// if we defined the graph wrongly we have to manually fix the definition.

		// It's like a SQL query: if the query is not syntactically valid
		// we have to rewrite it
		panic(err.Error())
	}

	// If here: our graph is syntatically valid.
	// We can now place it within a Session and execute it.

	var sess *tf.Session
	sess, err = tf.NewSession(graph, &tf.SessionOptions{})
	if err != nil {
		panic(err.Error())
	}

	// In order to use placeholders, we have to create the Tensors
	// containing the values to feed into the network
	var matrix, column *tf.Tensor

	// A = [ [1, 2], [-1, -2] ]
	if matrix, err = tf.NewTensor([2][2]int64{ {1, 2}, {-1, -2} }); err != nil {
		panic(err.Error())
	}
	// x = [ [10], [100] ]
	if column, err = tf.NewTensor([2][1]int64{ {10}, {100} }); err != nil {
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
		fmt.Println(result.Value().([][]int64))
	}
```

### Links
[Understanding TensorFlow and Go](https://pgaleone.eu/tensorflow/go/2017/05/29/understanding-tensorflow-using-go/) ||
[TensorFlow + Go, the gopher way](https://github.com/galeone/tfgo) ||
[TensorFlow Golang Package](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go) ||
[Enable TensorBoard for TensorFlow Go API](https://github.com/helinwang/tfsum) || 
[GoTensor](https://github.com/helinwang/gotensor)

### Notes
* [Understanding TensorFlow and Go](https://pgaleone.eu/tensorflow/go/2017/05/29/understanding-tensorflow-using-go/) 
  * A node is completely identified by the Scope in which it is defined.
  * The scope is the path that we have to follow from the root of the graph to reach the node.
  * There are 2 ways of defining nodes that execute the same operation: 
    * Define the operation in a different Scope (Go style)
    * or change the operation name (how Python automatically does or how we can do in C++)
  * The Go bindings have their own set of types, with a 1:1 mapping (almost complete) with the Go types
  * When we feed values into the graph we have to respect the mapping (for instance feed `int32` when defining `tf.Int32` placeholders). The same has to be done when fetching values from the graph.