pub const DenseLayer = struct {
    weights: [][]f32,
    biases: []f32,
    activation: fn (f32) f32,
};

fn dense(input: []f32, layer: DenseLayer) []f32 {
    // Perform matrix multiplication and apply activation.
    // Fully connected layer with matrix multiplication.
}
