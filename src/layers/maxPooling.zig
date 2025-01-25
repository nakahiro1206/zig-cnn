const Tensor3D = @import("../tensor/tensor3d.zig").Tensor3D;

pub const MaxPooling2DLayer = struct {
    pool_size: usize,
};

fn max_pooling2d(input: Tensor3D, pool_size: usize) Tensor3D {
    // Implement max pooling logic.
    // downsample the input tensor.
}
