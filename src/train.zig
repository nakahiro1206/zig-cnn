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
const eval = @import("evaluation.zig").eval;

pub fn train(trainDataPoints: DataPointSOA, testDataPoints: DataPointSOA) !void { // , __testDataPoints: DataPointSOA) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) expect(false) catch @panic("TEST FAIL");
    }

    const outputSize: usize = 10;

    // initial input shape = .{1, 28, 28}
    var conv1 = try Conv2DLayer.init(.{ 8, 1, 3, 3 }, allocator); // [channels_out, channels_in, height, width]
    defer conv1.dealloc();

    // input shape -> .{8, 26, 26}
    var pool = MaxPooling.init(.{ 2, 2 });

    // input shape -> .{8, 13, 13}
    var softmax = try Softmax.init(8 * 13 * 13, outputSize, allocator); // [input_size, output_size]
    defer softmax.dealloc();

    const epochs: usize = 1;
    const learningRate: f64 = 0.005;
    const batchSize = 1;
    const totalDataPoints = trainDataPoints.len();
    const batchCount: usize = totalDataPoints / batchSize;
    const inputSize = .{ batchSize, 1, 28, 28 };

    var stopwatch = perf.Stopwatch.new();

    var trainData = try NDArray(f64, 4).init(inputSize, allocator);
    defer trainData.deinit();
    var groundTruth = try NDArray(f64, 2).initWithValue(.{ batchSize, outputSize }, 0.0, allocator);
    defer groundTruth.deinit();

    var buf1: [5]u8 = undefined;
    var buf2: [5]u8 = undefined;

    for (0..epochs) |epochIndex| {
        std.debug.print("training epoch {} of {}\n", .{ epochIndex + 1, epochs });
        stopwatch.start();
        try trainDataPoints.shuffle(allocator);
        stopwatch.report("shuffle");
        var totalTrainAcc: f64 = 0.0;
        var totalTrainLoss: f64 = 0.0;

        for (0..batchCount) |batchIndex| {
            const start: usize = batchIndex * batchSize;
            const end: usize = start + batchSize;
            const str1 = try std.fmt.bufPrint(&buf1, "{d}", .{start});
            const str2 = try std.fmt.bufPrint(&buf2, "{d}", .{end});
            try stopwatch.reportForHeadding();
            std.debug.print("index range: {s: >5}..{s: >5} / {} | ", .{ str1, str2, totalDataPoints });

            const x_data = trainDataPoints.slice_x(start, batchSize);
            try trainData.setData(x_data);
            const y_data = trainDataPoints.slice_y(start, batchSize);
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
            const res = try eval(f4, groundTruth, allocator, true);
            totalTrainAcc += res.acc;
            totalTrainLoss += res.loss;
            var gradient = res.gradient;
            defer gradient.deinit();

            // backward
            var g1 = try softmax.backPropagation(gradient, f3, learningRate, allocator);
            defer g1.deinit();
            var g2 = try revert(g1, f2, allocator);
            defer g2.deinit();
            var g3 = try pool.backPropagation(g2, f1, allocator);
            defer g3.deinit();
            var g4 = try conv1.backPropagation(g3, trainData, learningRate, allocator);
            defer g4.deinit();
        }

        std.debug.print("Cumulative acc: {d}\n", .{totalTrainAcc / 60000.0});
        std.debug.print("Average loss: {d}\n", .{totalTrainLoss / 60000.0});
        stopwatch.report("trained");
        std.debug.print("evaluating...", .{});

        var testData = try NDArray(f64, 4).init(inputSize, allocator);
        defer testData.deinit();
        var testAnswer = try NDArray(f64, 2).initWithValue(.{ batchSize, outputSize }, 0.0, allocator);
        defer testAnswer.deinit();

        const testBatchCount = testDataPoints.len() / batchSize;
        try testDataPoints.shuffle(allocator);

        var totalLoss: f64 = 0.0;
        var totalAcc: f64 = 0.0;

        for (0..testBatchCount) |batchIndex| {
            const start: usize = batchIndex * batchSize;

            const x_data = testDataPoints.slice_x(start, batchSize);
            try testData.setData(x_data);
            const y_data = testDataPoints.slice_y(start, batchSize);
            try testAnswer.setData(y_data);

            // forward
            var f1 = try conv1.forward(testData, allocator); // f1: 3 * 2 * 2
            defer f1.deinit();
            var f2 = try pool.forward(f1, allocator); // f2: 3 * 1 * 1
            defer f2.deinit();
            var f3 = try flatten(f2, allocator); // f3: 3
            defer f3.deinit();
            var f4 = try softmax.forward(f3, allocator); // f4: 5
            defer f4.deinit();

            // evaluation
            const res = try eval(f4, testAnswer, allocator, false);
            var gradient = res.gradient;
            defer gradient.deinit();
            const loss = res.loss;
            const acc = res.acc;
            totalAcc += acc;
            totalLoss += loss;
        }

        totalAcc /= @as(f64, @floatFromInt(testBatchCount));
        totalLoss /= @as(f64, @floatFromInt(testBatchCount));

        var out1: [6]u8 = undefined;
        var out2: [6]u8 = undefined;
        const str1 = try std.fmt.bufPrint(&out1, "{d:.3}", .{totalLoss});
        const str2 = try std.fmt.bufPrint(&out2, "{d:.3}", .{totalAcc});
        std.debug.print("Done. loss: {s: >6}, acc: {s: >6}\n", .{ str1, str2 });
    }
}
