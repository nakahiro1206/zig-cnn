const std = @import("std");
const NDArray = @import("pblischak/zig-ndarray/ndarray.zig").NDArray;

pub const EvalResponse = struct {
    loss: f64,
    acc: f64,
    gradient: NDArray(f64, 2),
};

pub fn eval(prediction: NDArray(f64, 2), groundTruth: NDArray(f64, 2), allocator: std.mem.Allocator, logging: bool) !EvalResponse {
    // prediction: [batchSize, outputSize]
    const outputSize = prediction.shape[1];
    const batchSize = prediction.shape[0];

    var gradient = try NDArray(f64, 2).initWithValue(.{ batchSize, outputSize }, 0.0, allocator);

    var lossTotal: f64 = 0.0;
    var accTotal: f64 = 0.0;
    const batchSizef64: f64 = @as(f64, @floatFromInt(batchSize));

    const epsilon: f64 = 1e-10; // Small value to prevent division by zero
    for (0..batchSize) |batchIndex| {
        var answerIndex: usize = 0;
        var predictedIndex: usize = 0;
        for (0..outputSize) |outputIndex| {
            if (prediction.atConst(.{ batchIndex, outputIndex }) > prediction.atConst(.{ batchIndex, predictedIndex })) {
                predictedIndex = outputIndex;
            }
            if (groundTruth.atConst(.{ batchIndex, outputIndex }) > groundTruth.atConst(.{ batchIndex, answerIndex })) {
                answerIndex = outputIndex;
            }
        }
        lossTotal += -1 * @log(prediction.atConst(.{ batchIndex, answerIndex }) + epsilon);
        if (answerIndex == predictedIndex) {
            accTotal += 1.0;
        }

        gradient.setAt(.{ batchIndex, answerIndex }, -1 / (prediction.atConst(.{ batchIndex, answerIndex }) + epsilon));
    }

    if (logging) {
        var buf1: [6]u8 = undefined;
        var buf2: [6]u8 = undefined;
        const str1 = try std.fmt.bufPrint(&buf1, "{d:.3}", .{lossTotal / batchSizef64});
        const str2 = try std.fmt.bufPrint(&buf2, "{d:.3}", .{accTotal / batchSizef64});
        std.debug.print("loss: {s: >6}, acc: {s: >6}\n", .{ str1, str2 });
    }

    return EvalResponse{ .loss = lossTotal / batchSizef64, .acc = accTotal / batchSizef64, .gradient = gradient };
}
