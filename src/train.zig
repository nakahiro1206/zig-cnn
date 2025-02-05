const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const perf = @import("albert-yu/mnist/performance.zig");

const DataPointSOA = @import("albert-yu/mnist/datapoint.zig").DataPointSOA;
const Conv2DLayer = @import("layers/conv2d.zig").Conv2DLayer;
const Softmax = @import("layers/softmax.zig").Softmax;
const Flatten = @import("layers/flatten.zig").Flatten;
const MaxPooling = @import("layers/maxPooling.zig").MaxPoolingLayer;
const NDArray = @import("pblischak/zig-ndarray/ndarray.zig").NDArray;
const flatten = @import("layers/flatten.zig").flatten;
const revert = @import("layers/flatten.zig").revert;

pub fn train(trainDataPoints: DataPointSOA) !void { // , __testDataPoints: DataPointSOA) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) expect(false) catch @panic("TEST FAIL");
    }

    const inputSize = .{ 28, 28 };
    const outputSize: usize = 10;

    // input size: 28 * 28
    var conv1 = try Conv2DLayer.init(8, .{ 3, 3 }, allocator); // filter size: 8 * 3 * 3
    defer conv1.dealloc();
    // input size -> 8 * 26 * 26
    var pool = MaxPooling.init(.{ 2, 2 });
    // input size -> 8 * 13 * 13
    // faltten: input -> 8 * 13 * 13
    var softmax = try Softmax.init(.{8 * 13 * 13}, outputSize, allocator);
    defer softmax.dealloc();
    // output size -> 10

    const epochs: usize = 100;
    const learningRate: f64 = 0.0005;
    const batchSize = 1;
    const batchCount: usize = trainDataPoints.len() / batchSize;

    var stopwatch = perf.Stopwatch.new();

    var trainData = try NDArray(f64, 2).init(inputSize, allocator);
    defer trainData.deinit();
    var groundTruth = try NDArray(f64, 1).initWithValue(.{outputSize}, 0.0, allocator);
    defer groundTruth.deinit();
    // var error = try NDArray(f64, 1).initWithValue(.{ outputSize }, 0.0, allocator);
    // defer error.deinit();

    for (0..epochs) |epochIndex| {
        std.debug.print("training epoch {} of {}\n", .{ epochIndex + 1, epochs });
        stopwatch.start();
        try trainDataPoints.shuffle(allocator);
        stopwatch.report("shuffle");
        // const scalar: f64 = learningRate / @floatFromInt(batchSize); // for what??

        for (0..batchCount) |batchIndex| {
            const start: usize = batchIndex * batchSize;
            const end: usize = start + batchSize;

            for (start..end) |index| {
                const x_data = trainDataPoints.x_at(index);
                try trainData.setData(x_data);
                const y_data = trainDataPoints.y_at(index);
                try groundTruth.setData(y_data);

                // forward
                var f1 = try conv1.forward(trainData, allocator); // f1: 3 * 2 * 2
                defer f1.deinit();
                var f2 = try pool.forward(f1, allocator); // f2: 3 * 1 * 1
                defer f2.deinit();
                var f3 = try flatten(f2, allocator); // f3: 3
                defer f3.deinit();
                var f4 = try softmax.forward(f3, allocator); // f4: 5
                defer f4.deinit();

                // evaluation
                var predictedIndex: usize = 0;
                var answerIndex: usize = 0;
                for (0..outputSize) |i| {
                    if (f4.at(.{i}) > f4.at(.{predictedIndex})) {
                        predictedIndex = i;
                    }
                    if (groundTruth.at(.{i}) > groundTruth.at(.{answerIndex})) {
                        answerIndex = i;
                    }
                }

                const epsilon: f64 = 1e-10; // Small value to prevent division by zero
                const loss = -1 * @log(f4.at(.{answerIndex})) + epsilon;
                const acc: bool = answerIndex == predictedIndex;
                if (index % 100 == 0) {
                    stopwatch.report("mid-report");
                    // std.debug.print("index: {d}, {d} / {d}, {any}\n", .{index, batchIndex, batchCount, f1.items});
                    // std.debug.print("f4 val: {d}, ", .{f4.at(.{answerIndex})});
                    std.debug.print("loss: {d}, acc: {any}\n", .{ loss, acc });
                }
                var gradient = try NDArray(f64, 1).initWithValue(.{outputSize}, 0.0, allocator);
                defer gradient.deinit();
                gradient.setAt(.{answerIndex}, -1 / (f4.at(.{answerIndex}) + epsilon));

                // backward
                var g1 = try softmax.backPropagation(gradient, f3, learningRate, allocator);
                defer g1.deinit();
                var g2 = try revert(g1, f2, allocator);
                defer g2.deinit();
                var g3 = try pool.backPropagation(g2, f1, allocator);
                defer g3.deinit();
                try conv1.backPropagation(g3, trainData, learningRate, allocator);

                // if (index % 100 == 0) {
                //     std.debug.print("grad: {any}, g1: {any}, g2: {any}, g3: {any}\n", .{ gradient.items, g1.items, g2.items, g3.items });
                // }
            }
        }

        stopwatch.report("trained");
        std.debug.print("evaluating...", .{});

        //         var i: usize = 0;
        //         var correct: usize = 0;
        //         while (i < test_data.len()) : (i += 1) {
        //             const x_data = test_data.x_at(i);
        //             x.set_data(x_data);
        //             const y_data = test_data.y_at(i);

        //             const activations1 = try layer1.forward(allocator, x, maths.sigmoid);
        //             const activations2 = try layer2.forward(allocator, activations1, maths.sigmoid);

        //             const expected = find_max_index(y_data);
        //             const result_data = try activations2.dump(allocator);
        //             defer allocator.free(result_data);
        //             const actual = find_max_index(result_data);
        //             if (expected == actual) {
        //                 correct += 1;
        //             }
        //         }
        //         std.debug.print("done. {}/{} correct\n", .{ correct, test_data.len() });
    }
}

//     const HIDDEN_LAYER_SIZE = 30;
//     const ETA = 0.05;
//     const EPOCHS = 100;

//     var layer1 = try layers.Layer(image_size, HIDDEN_LAYER_SIZE).alloc(allocator);
//     defer layer1.dealloc(allocator);
//     var layer2 = try layers.Layer(HIDDEN_LAYER_SIZE, DIGITS).alloc(allocator);
//     defer layer2.dealloc(allocator);

//     layer1.init_randn();
//     layer2.init_randn();

//     const BATCH_SIZE = 10;

//     const batch_count = train_data_points.len() / BATCH_SIZE;

//     var stopwatch = perf.Stopwatch.new();
//     var epoch_index: usize = 0;
//     while (epoch_index < EPOCHS) : (epoch_index += 1) {
//         std.debug.print("training epoch {} of {}\n", .{ epoch_index + 1, EPOCHS });
//         stopwatch.start();
//         try train_data_points.shuffle(allocator);
//         stopwatch.report("shuffle");
//         var batch_index: usize = 0;
//         const scalar: comptime_float = ETA / @as(f64, @floatFromInt(BATCH_SIZE));

//         var x = linalg.Matrix.new(image_size, 1);
//         try x.alloc(allocator);
//         defer x.dealloc(allocator);
//         var y = linalg.Matrix.new(DIGITS, 1);
//         try y.alloc(allocator);
//         defer y.dealloc(allocator);

//         var err = linalg.Matrix.new(DIGITS, 1);
//         try err.alloc(allocator);
//         defer err.dealloc(allocator);

//         while (batch_index < batch_count) : (batch_index += 1) {
//             var i = batch_index * BATCH_SIZE;
//             const end = i + BATCH_SIZE;

//             while (i < end) : (i += 1) {
//                 const x_data = train_data_points.x_at(i);
//                 // console_print_image(x_data, 28);
//                 x.set_data(x_data);
//                 const y_data = train_data_points.y_at(i);
//                 y.set_data(y_data);

//                 // forward
//                 const activations1 = try layer1.forward(allocator, x, maths.sigmoid);
//                 const activations2 = try layer2.forward(allocator, activations1, maths.sigmoid);
//                 activations2.sub(y, &err);

//                 // backward
//                 var grad2 = try layer2.backward(allocator, err, maths.sigmoid_prime);

//                 var err_inner = try layer2.weights.t_alloc(allocator);
//                 defer err_inner.dealloc(allocator);

//                 var err_inner2 = try err_inner.mul_alloc(allocator, grad2.biases);
//                 defer err_inner2.dealloc(allocator);
//                 var grad1 = try layer1.backward(allocator, err_inner2, maths.sigmoid_prime);

//                 // apply gradients
//                 grad1.biases.scale(scalar);
//                 layer1.biases.sub(grad1.biases, &layer1.biases);
//                 grad1.weights.scale(scalar);
//                 layer1.weights.sub(grad1.weights, &layer1.weights);

//                 grad2.biases.scale(scalar);
//                 layer2.biases.sub(grad2.biases, &layer2.biases);
//                 grad2.weights.scale(scalar);
//                 layer2.weights.sub(grad2.weights, &layer2.weights);
//             }
//         }
//         stopwatch.report("trained");
//         std.debug.print("evaluating...", .{});

//         var i: usize = 0;
//         var correct: usize = 0;
//         while (i < test_data.len()) : (i += 1) {
//             const x_data = test_data.x_at(i);
//             x.set_data(x_data);
//             const y_data = test_data.y_at(i);

//             const activations1 = try layer1.forward(allocator, x, maths.sigmoid);
//             const activations2 = try layer2.forward(allocator, activations1, maths.sigmoid);

//             const expected = find_max_index(y_data);
//             const result_data = try activations2.dump(allocator);
//             defer allocator.free(result_data);
//             const actual = find_max_index(result_data);
//             if (expected == actual) {
//                 correct += 1;
//             }
//         }
//         std.debug.print("done. {}/{} correct\n", .{ correct, test_data.len() });
//     }

//     std.debug.print("done.\n", .{});
// }
