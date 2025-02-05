const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;

const Conv2DLayer = @import("layers/conv2d.zig").Conv2DLayer;
const Softmax = @import("layers/softmax.zig").Softmax;
const Flatten = @import("layers/flatten.zig").Flatten;
const MaxPooling = @import("layers/maxPooling.zig").MaxPoolingLayer;
const NDArray = @import("pblischak/zig-ndarray/ndarray.zig").NDArray;
const flatten = @import("layers/flatten.zig").flatten;
const revert = @import("layers/flatten.zig").revert;

test "cnn training" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) expect(false) catch @panic("TEST FAIL");
    }

    const outputSize: usize = 5;
    const epochs: usize = 50;
    const learningRate: f64 = 0.05;

    // input size: 4 * 4
    var conv1 = try Conv2DLayer.init(3, .{ 3, 3 }, allocator); // 3 * 3 * 3
    defer conv1.dealloc();
    // input size -> 3 * 2 * 2
    var pool = MaxPooling.init(.{ 2, 2 }); // 3 * 2 * 2
    // input size -> 3 * 1 * 1
    // faltten: input -> 3
    var softmax = try Softmax.init(.{3}, 5, allocator);
    defer softmax.dealloc();
    // output size -> 5

    for (0..epochs) |_| {
        // var input = try NDArray(f64, 2).fromCopiedSlice(.{ 4, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }, allocator); // 4*4
        var input = try NDArray(f64, 2).initWithValue(.{ 4, 4 }, 0.5, allocator);
        defer input.deinit();
        const label: usize = 3;

        // var layer: Conv2DLayer = try Conv2DLayer.init(3, .{ 3, 3 }, allocator);
        // defer layer.dealloc();

        // var input = try NDArray(f64, 2).initWithValue(.{ 10, 10 }, 0.5, allocator);
        // defer input.deinit();

        // var output = try layer.forward(input, allocator);
        // defer output.deinit();

        var f1 = try conv1.forward(input, allocator); // f1: 3 * 2 * 2
        defer f1.deinit();
        var f2 = try pool.forward(f1, allocator); // f2: 3 * 1 * 1
        defer f2.deinit();
        var f3 = try flatten(f2, allocator); // f3: 3
        defer f3.deinit();
        var f4 = try softmax.forward(f3, allocator); // f4: 5
        defer f4.deinit();

        const loss = -1 * @log(f4.at(.{label}));
        var maxIndex: usize = 0;
        for (f4.items, 0..) |item, index| {
            if (item > f4.at(.{maxIndex})) {
                maxIndex = index;
            }
        }
        const acc: bool = label == maxIndex;
        std.debug.print("loss: {d}, acc: {any}\n", .{ loss, acc });
        var gradient = try NDArray(f64, 1).initWithValue(.{outputSize}, 0.0, allocator);
        defer gradient.deinit();
        gradient.setAt(.{label}, -1 / f4.at(.{label}));

        var g1 = try softmax.backPropagation(gradient, f3, learningRate, allocator);
        defer g1.deinit();
        var g2 = try revert(g1, f2, allocator);
        defer g2.deinit();
        var g3 = try pool.backPropagation(g2, f1, allocator);
        defer g3.deinit();
        try conv1.backPropagation(g3, input, learningRate, allocator);
    }

    // def train(im, label, lr=.005):
    //   '''
    //   Completes a full training step on the given image and label.
    //   Returns the cross-entropy loss and accuracy.
    //   - image is a 2d numpy array
    //   - label is a digit
    //   - lr is the learning rate
    //   '''
    //   # Forward
    //   out, loss, acc = forward(im, label)

    //   # Calculate initial gradient
    //   gradient = np.zeros(10)
    //   gradient[label] = -1 / out[label]

    //   # Backprop
    //   gradient = softmax.backprop(gradient, lr)
    //   gradient = pool.backprop(gradient)
    //   gradient = conv.backprop(gradient, lr)

    //   return loss, acc
}
