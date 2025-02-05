const std = @import("std");
const expect = std.testing.expect;
const NDArray = @import("../pblischak/zig-ndarray/ndarray.zig").NDArray;
const Softmax = @import("softmax.zig").Softmax;

pub fn softmaxTest() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) expect(false) catch @panic("TEST FAIL");
    }

    var layer = try Softmax.init(.{8}, 5, allocator);
    defer layer.dealloc();
    // try Softmax(.{2,2,2}, .{5}) wanna specify I/O.

    var input = try NDArray(f64, 1).initWithValue(.{8}, 0.5, allocator);
    defer input.deinit();

    // forward
    var output = try layer.forward(input, allocator);
    defer output.deinit();

    // backprop
    var output2 = try layer.backPropagation(output, input, -0.05, allocator);
    defer output2.deinit();

    std.debug.print("input: {any}\n", .{input.items});
    std.debug.print("forward: {any}\n", .{output.items});
    std.debug.print("backprop: {any}\n", .{output2.items});

    // try expect(output.shape[0] == 1 and output.shape[1] == 2 and output.shape[2] == 2);
}
