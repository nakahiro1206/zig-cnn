const std = @import("std");
const expect = std.testing.expect;
const Allocator = std.mem.Allocator;

const NDArray = @import("../../pblischak/zig-ndarray/ndarray.zig").NDArray;
const flatten = @import("../flatten.zig").flatten;
const revert = @import("../flatten.zig").revert;

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
