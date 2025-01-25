const std = @import("std");
const expect = std.testing.expect;

const Conv2DLayer = @import("layers/conv2d.zig").Conv2DLayer;
const NDArray = @import("pblischak/zig-ndarray/ndarray.zig").NDArray;
// test "Conv2DLayer" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer {
//         const deinit_status = gpa.deinit();
//         //fail test; can't try in defer as defer is executed after we return
//         if (deinit_status == .leak) expect(false) catch @panic("TEST FAIL");
//     }

//     var layer: Conv2DLayer = try Conv2DLayer.init(3, .{ 3, 3 }, allocator);
//     defer layer.dealloc();

//     layer.print();

//     var input = try NDArray(f64, 2).initWithValue(.{ 10, 10 }, 0.5, allocator);
//     defer input.deinit();
//     var output = try layer.forward(input, allocator);
//     defer output.deinit();

//     layer.print();
//     std.debug.print("output: {any}\n", .{output.shape});

//     try expect(layer.filters_num == 3);
//     try expect(layer.filter_size[0] == 3 and layer.filter_size[1] == 3);

//     try expect(output.shape[0] == 8 and output.shape[1] == 8);
// }
test "Conv2DLayer forward pass" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) expect(false) catch @panic("TEST FAIL");
    }

    var layer: Conv2DLayer = try Conv2DLayer.init(3, .{ 3, 3 }, allocator);
    defer layer.dealloc();

    var input = try NDArray(f64, 2).initWithValue(.{ 10, 10 }, 0.5, allocator);
    defer input.deinit();

    var output = try layer.forward(input, allocator);
    defer output.deinit();

    std.debug.print("output: {any}\n", .{output});
    std.debug.print("filters: {any}\n", .{layer.filters});
    std.debug.print("last input: {any}\n", .{layer.lastInput});

    try expect(layer.filters_num == 3);
    try expect(layer.filter_size[0] == 3 and layer.filter_size[1] == 3);
    try expect(output.shape[0] == 8 and output.shape[1] == 8);
}

test "Conv2DLayer memory leak check" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) expect(false) catch @panic("TEST FAIL");
    }

    var layer: Conv2DLayer = try Conv2DLayer.init(3, .{ 3, 3 }, allocator);
    defer layer.dealloc();

    var input = try NDArray(f64, 2).initWithValue(.{ 10, 10 }, 0.5, allocator);
    defer input.deinit();

    var output = try layer.forward(input, allocator);
    defer output.deinit();

    try expect(layer.filters_num == 3);
    try expect(layer.filter_size[0] == 3 and layer.filter_size[1] == 3);
    try expect(output.shape[0] == 8 and output.shape[1] == 8);
}

test "back propagation test" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) expect(false) catch @panic("TEST FAIL");
    }

    var layer: Conv2DLayer = try Conv2DLayer.init(3, .{ 3, 3 }, allocator);
    defer layer.dealloc();

    var input = try NDArray(f64, 2).initWithValue(.{ 10, 10 }, 0.5, allocator);
    defer input.deinit();

    var output = try layer.forward(input, allocator);
    defer output.deinit();

    std.debug.print("Before filters: {any}\n", .{layer.filters});

    // use output for dummy `lossGradientForOutput`
    try layer.backPropagation(output, 0.05, allocator);

    std.debug.print("After filters: {any}\n", .{layer.filters});

    try expect(layer.filters_num == 3);
    try expect(layer.filter_size[0] == 3 and layer.filter_size[1] == 3);
    try expect(output.shape[0] == 8 and output.shape[1] == 8);
}
