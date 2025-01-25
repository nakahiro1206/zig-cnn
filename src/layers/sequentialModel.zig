const Conv2DLayer = @import("Conv2DLayer.zig").Conv2DLayer;
const MaxPooling2DLayer = @import("MaxPooling2DLayer.zig").MaxPooling2DLayer;
const DenseLayer = @import("DenseLayer.zig").DenseLayer;

pub const SequentialModel = struct {
    conv2d_layers: []Conv2DLayer,
    maxpool_layers: []MaxPooling2DLayer,
    dense_layers: []DenseLayer,
};

fn forward_propagate(model: SequentialModel, input: Tensor3D) []f32 {
    var output = input;
    for (model.conv2d_layers) |layer| output = conv2d(output, layer);
    for (model.maxpool_layers) |layer| output = max_pooling2d(output, layer.pool_size);
    // Flatten and pass through dense layers.
}

fn backward_propagate(model: SequentialModel, input: Tensor3D) []f32 {
    // Implement backpropagation logic.
    // Calculate gradients and update weights.
}

fn cross_entropy_loss(predicted: []f32, actual: []f32) f32 {
    // Compute cross-entropy.
}

fn update_weights(layer: DenseLayer, gradients: [][]f32, learning_rate: f32) void {
    // Update weights and biases.
}
