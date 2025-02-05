const std = @import("std");
const expect = std.testing.expect;

const Conv2DLayer = @import("layers/conv2d.zig").Conv2DLayer;
const MaxPoolingLayer = @import("layers/maxPooling.zig").MaxPoolingLayer;
const NDArray = @import("pblischak/zig-ndarray/ndarray.zig").NDArray;

test "conv2d test" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) expect(false) catch @panic("TEST FAIL");
    }

    // layers
    var conv2DLayer: Conv2DLayer = try Conv2DLayer.init(3, .{ 3, 3 }, allocator);
    defer conv2DLayer.dealloc();
    const poolingLayer: MaxPoolingLayer = MaxPoolingLayer.init(.{ 2, 2 });

    var input = try NDArray(f64, 2).initWithValue(.{ 10, 10 }, 0.5, allocator);
    defer input.deinit();

    // forward
    var forward1 = try conv2DLayer.forward(input, allocator);
    defer forward1.deinit();
    std.debug.print("Before filters: {any}\n", .{forward1.filters});

    var forward2 = try poolingLayer.forward(forward1, allocator);
    defer forward2.deinit();

    // back propagation
    var back1 = try poolingLayer.backPropagation(forward2, forward1, 0.05, allocator);

    // use output for dummy `lossGradientForOutput(gradient, lastInputUseAsForwered)`
    try layer.backPropagation(output, input, 0.05, allocator);

    std.debug.print("After filters: {any}\n", .{layer.filters});

    try expect(layer.filters_num == 3);
    try expect(layer.filter_size[0] == 3 and layer.filter_size[1] == 3);
    try expect(output.shape[0] == 8 and output.shape[1] == 8);
}
