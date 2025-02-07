const std = @import("std");
const Allocator = std.mem.Allocator;

const NDArray = @import("../pblischak/zig-ndarray/ndarray.zig").NDArray;

pub fn flatten(input: NDArray(f64, 4), allocator: Allocator) !NDArray(f64, 2) {
    // input: [batch_size, channels, height, width]
    // ouutput: [batch_size, total_length]
    //   Performs a forward pass of the flatten layer using the given input.
    //   Returns a 1d numpy array containing the respective probability values.
    //   - input can be any array with any dimensions.
    const batchSize = input.shape[0];
    const inputLength: usize = input.shape[1] * input.shape[2] * input.shape[3];
    const flattenArray = try NDArray(f64, 2).fromCopiedSlice(.{ batchSize, inputLength }, input.items, allocator);
    return flattenArray;
}

pub fn revert(output: NDArray(f64, 2), lastInput: NDArray(f64, 4), allocator: Allocator) !NDArray(f64, 4) {
    //   Performs a backward pass of the flatten layer.
    //   Returns a 3d numpy array with the same dimensions as the input.
    //   - input can be any array with any dimensions.
    const shape = lastInput.shape;
    const backprop = try NDArray(f64, 4).fromCopiedSlice(shape, output.items, allocator);
    return backprop;
}

const expect = std.testing.expect;
fn isSameShape(N: comptime_int, in: [N]usize, out: [N]usize) bool {
    for (0..N) |i| {
        if (in[i] != out[i]) return false;
    }
    return true;
}
pub fn flattenTest() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) expect(false) catch @panic("TEST FAIL");
    }

    var input = try NDArray(f64, 4).initWithValue(.{ 2, 2, 2, 2 }, 0.0, allocator);
    defer input.deinit();
    input.setAt(.{ 0, 1, 0, 1 }, 5.0); // should indexed {0, 5}

    // need to update!!
    var forward = try flatten(input, allocator);
    defer forward.deinit();

    try expect(isSameShape(2, forward.shape, .{ 2, 8 }));
    try expect(forward.at(.{ 0, 5 }) == 5.0);

    var backPropagation = try revert(forward, input, allocator);
    defer backPropagation.deinit();

    try expect(isSameShape(4, backPropagation.shape, input.shape));
    try expect(backPropagation.at(.{ 0, 1, 0, 1 }) == 5.0);
}
