const std = @import("std");
const expect = std.testing.expect;

const Conv2DLayer = @import("../conv2d.zig").Conv2DLayer;
const NDArray = @import("../../pblischak/zig-ndarray/ndarray.zig").NDArray;

pub fn Conv2DTest() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) expect(false) catch @panic("TEST FAIL");
    }

    // outputChannel, inputChannel, height, width
    var layer: Conv2DLayer = try Conv2DLayer.init(.{ 32, 1, 3, 3 }, allocator);
    defer layer.dealloc();

    var input = try NDArray(f64, 4).initWithValue(.{ 128, 1, 10, 10 }, 0.5, allocator);
    defer input.deinit();

    var output = try layer.forward(input, allocator);
    defer output.deinit();

    try expect(layer.filterShape[0] == 32 and layer.filterShape[1] == 1 and layer.filterShape[2] == 3 and layer.filterShape[3] == 3); // filter shape should be 3 * 3 * 3
    try expect(output.shape[0] == 128 and output.shape[1] == 32 and output.shape[2] == 8 and output.shape[3] == 8); // output shape should be 3 * 8 * 8

    // use output for dummy `lossGradientForOutput(gradient, lastInputUseAsForwered)`
    var gradient = try layer.backPropagation(output, input, 0.05, allocator);
    defer gradient.deinit();

    try expect(gradient.shape[0] == 128 and gradient.shape[1] == 1 and gradient.shape[2] == 10 and gradient.shape[3] == 10); // gradient shape should be 128 * 1 * 10 * 10
}
