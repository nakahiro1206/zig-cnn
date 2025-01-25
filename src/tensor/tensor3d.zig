pub const Tensor3D = struct {
    width: usize,
    height: usize,
    channels: usize,
    data: [][][f32],
};
