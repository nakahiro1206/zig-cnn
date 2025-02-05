const std = @import("std");
const expect = std.testing.expect;
const MaxPoolingLayer = @import("maxPooling.zig").MaxPoolingLayer;
const NDArray = @import("../pblischak/zig-ndarray/ndarray.zig").NDArray;

fn isSameShape(N: comptime_int, in: [N]usize, out: [N]usize) bool {
    for (0..N) |i| {
        if (in[i] != out[i]) return false;
    }
    return true;
}

pub fn maxPoolingTest() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) expect(false) catch @panic("TEST FAIL");
    }

    const layer: MaxPoolingLayer = MaxPoolingLayer.init(.{ 2, 2 });

    var input = try NDArray(f64, 4).initWithValue(.{ 128, 1, 4, 3 }, 0.5, allocator);
    defer input.deinit();
    input.setAt(.{ 0, 0, 0, 0 }, 10);
    input.setAt(.{ 0, 0, 3, 2 }, 2);

    // forward
    var output = try layer.forward(input, allocator);
    defer output.deinit();

    try expect(output.atConst(.{ 0, 0, 0, 0 }) == 10);
    try expect(output.atConst(.{ 0, 0, 1, 1 }) == 2);
    try expect(isSameShape(4, output.shape, .{ 128, 1, 2, 2 }) == true);

    // backprop
    var output2 = try layer.backPropagation(output, input, allocator);
    defer output2.deinit();

    try expect(isSameShape(4, input.shape, output2.shape) == true);
}
