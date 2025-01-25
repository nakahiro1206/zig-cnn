const model = SequentialModel{
    .conv2d_layers = &[_]Conv2DLayer{
        Conv2DLayer{ .filters = ..., .bias = ..., .activation = relu },
        Conv2DLayer{ .filters = ..., .bias = ..., .activation = relu },
    },
    .maxpool_layers = &[_]MaxPooling2DLayer{
        MaxPooling2DLayer{ .pool_size = 2 },
    },
    .dense_layers = &[_]DenseLayer{
        DenseLayer{ .weights = ..., .biases = ..., .activation = relu },
        DenseLayer{ .weights = ..., .biases = ..., .activation = softmax },
    },
};
