const std = @import("std");

pub const Layer = struct {
    weights: [][]f32, // 2D array for weights.
    biases: []f32, // 1D array for biases.
};

pub const CNN = struct {
    layers: []Layer,
};

fn train(model: CNN, data: [][]f32, labels: []f32, epochs: usize) void {
    for (0..epochs) |epoch| {
        // Forward propagation.
        // Compute loss.
        // Backpropagation.
        // Update weights.
    }
}
