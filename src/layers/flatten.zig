const std = @import("std");
const Allocator = std.mem.Allocator;

const NDArray = @import("../pblischak/zig-ndarray/ndarray.zig").NDArray;

pub fn flatten(input: NDArray(f64, 3), allocator: Allocator) !NDArray(f64, 1) {
    //   Performs a forward pass of the flatten layer using the given input.
    //   Returns a 1d numpy array containing the respective probability values.
    //   - input can be any array with any dimensions.
    const len = input.items.len;
    const flattenArray = try NDArray(f64, 1).fromCopiedSlice(.{len}, input.items, allocator);

    return flattenArray;
}

pub fn revert(output: NDArray(f64, 1), lastInput: NDArray(f64, 3), allocator: Allocator) !NDArray(f64, 3) {
    //   Performs a backward pass of the flatten layer.
    //   Returns a 3d numpy array with the same dimensions as the input.
    //   - input can be any array with any dimensions.
    const shape = lastInput.shape;
    const backprop = try NDArray(f64, 3).fromCopiedSlice(shape, output.items, allocator);
    return backprop;
}

const expect = std.testing.expect;
pub fn flattenTest() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) expect(false) catch @panic("TEST FAIL");
    }

    var input = try NDArray(f64, 3).initWithValue(.{ 2, 2, 2 }, 0.0, allocator);
    defer input.deinit();

    var forward = try flatten(input, allocator);
    defer forward.deinit();
    var backPropagation = try revert(forward, input, allocator);
    defer backPropagation.deinit();

    std.debug.print("init filters: {any}\n", .{input});
    std.debug.print("Before filters: {any}\n", .{forward});
    std.debug.print("After filters: {any}\n", .{backPropagation});

    try std.testing.expectEqual(forward.shape, .{8});
    try std.testing.expectEqual(backPropagation.shape, .{ 2, 2, 2 });
    try std.testing.expectEqual(forward.items.len, 8);
    try std.testing.expectEqual(backPropagation.items.len, 8);
}
