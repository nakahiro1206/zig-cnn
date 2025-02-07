const std = @import("std");
const expect = std.testing.expect;
const NDArray = @import("../../pblischak/zig-ndarray/ndarray.zig").NDArray;
const Softmax = @import("../softmax.zig").Softmax;

fn isSameShape(N: comptime_int, in: [N]usize, out: [N]usize) bool {
    for (0..N) |i| {
        if (in[i] != out[i]) return false;
    }
    return true;
}
pub fn softmaxTest() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) expect(false) catch @panic("TEST FAIL");
    }

    var layer = try Softmax.init(50, 5, allocator);
    defer layer.dealloc();

    var input = try NDArray(f64, 2).initWithValue(.{ 128, 50 }, 0.5, allocator);
    defer input.deinit();

    // forward
    var output = try layer.forward(input, allocator);
    defer output.deinit();

    try expect(isSameShape(2, output.shape, .{ 128, 5 }));

    // backprop
    var output2 = try layer.backPropagation(output, input, -0.05, allocator);
    defer output2.deinit();

    try expect(isSameShape(2, input.shape, output2.shape));

    // std.debug.print("input: {any}\n", .{input.items});
    // std.debug.print("forward: {any}\n", .{output.items});
    // std.debug.print("backprop: {any}\n", .{output2.items});
}
