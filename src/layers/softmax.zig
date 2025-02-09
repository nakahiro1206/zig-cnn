const std = @import("std");
const Allocator = std.mem.Allocator;
const NDArray = @import("../pblischak/zig-ndarray/ndarray.zig").NDArray;
const RndGen = std.rand.DefaultPrng;

pub const Softmax = struct {
    // promise: input should be flatten.
    inputChannel: usize,
    outputChannel: usize,
    weights: NDArray(f64, 2), // [inputChannel, outputChannel]
    bias: NDArray(f64, 1), // [outputChannel]

    const Self = @This();

    pub fn init(inputChannel: usize, outputChannel: usize, allocator: std.mem.Allocator) !Self {
        var weights: NDArray(f64, 2) = try NDArray(f64, 2).init(.{ inputChannel, outputChannel }, allocator);
        const bias: NDArray(f64, 1) = try NDArray(f64, 1).initWithValue(.{outputChannel}, 0, allocator);

        var prng = RndGen.init(@as(u64, @intCast(std.time.milliTimestamp())));
        const rand = prng.random();

        for (0..inputChannel) |i| {
            for (0..outputChannel) |o| {
                weights.setAt(.{ i, o }, (rand.float(f64) * 2 - 1) * std.math.sqrt(2.0 / @as(f64, @floatFromInt(inputChannel))));
            }
        }

        return Self{
            .inputChannel = inputChannel,
            .weights = weights,
            .bias = bias,
            .outputChannel = outputChannel,
        };
    }

    pub fn dealloc(self: *Self) void {
        self.weights.deinit();
        self.bias.deinit();
    }

    pub fn forward(self: *Self, input: NDArray(f64, 2), allocator: Allocator) !NDArray(f64, 2) {
        //   Performs a forward pass of the softmax layer using the given input.
        //   Returns a 1d numpy array containing the respective probability values.
        //   - input can be any array with any dimensions.
        const batchSize = input.shape[0];
        var softmax = try NDArray(f64, 2).initWithValue(.{ batchSize, self.outputChannel }, 0.0, allocator);

        // // Z = XW + B
        // var totalPerBatch = try NDArray(f64, 1).initWithValue(.{batchSize}, 0.0, allocator);
        // defer totalPerBatch.deinit();
        // for (0..batchSize) |batchIdx| {
        //     var sumTotal: f64 = 0.0;
        //     for (0..self.outputChannel) |outputIndex| {
        //         var sum: f64 = 0.0;
        //         for (0..self.inputChannel) |inputChannelIndex| {
        //             sum += input.atConst(.{ batchIdx, inputChannelIndex }) * self.weights.at(.{ inputChannelIndex, outputIndex });
        //         }
        //         sum += self.bias.at(.{outputIndex});
        //         softmax.setAt(.{ batchIdx, outputIndex }, sum);
        //         sumTotal += sum;
        //     }
        //     totalPerBatch.setAt(.{batchIdx}, sumTotal);
        // }

        // for (0..batchSize) |batchIdx| {
        //     for (0..self.outputChannel) |outputIndex| {
        //         softmax.setAt(.{ batchIdx, outputIndex }, @exp(softmax.at(.{ batchIdx, outputIndex }) - totalPerBatch.at(.{batchIdx})));
        //     }
        // }

        // Step 1: Compute Z = XW + B
        var maxPerBatch = try NDArray(f64, 1).initWithValue(.{batchSize}, -std.math.inf(f64), allocator);
        defer maxPerBatch.deinit();

        for (0..batchSize) |batchIdx| {
            for (0..self.outputChannel) |outputIndex| {
                var sum: f64 = 0.0;
                for (0..self.inputChannel) |inputChannelIndex| {
                    sum += input.atConst(.{ batchIdx, inputChannelIndex }) * self.weights.at(.{ inputChannelIndex, outputIndex });
                }
                sum += self.bias.at(.{outputIndex});
                softmax.setAt(.{ batchIdx, outputIndex }, sum);

                // Track max value per batch for numerical stability
                if (sum > maxPerBatch.at(.{batchIdx})) {
                    maxPerBatch.setAt(.{batchIdx}, sum);
                }
            }
        }

        // Step 2: Compute exp(Z - max(Z)) and sum over output channels
        var sumExpPerBatch = try NDArray(f64, 1).initWithValue(.{batchSize}, 0.0, allocator);
        defer sumExpPerBatch.deinit();

        for (0..batchSize) |batchIdx| {
            for (0..self.outputChannel) |outputIndex| {
                const stabilized = softmax.at(.{ batchIdx, outputIndex }) - maxPerBatch.at(.{batchIdx});
                const expVal = @exp(stabilized);
                softmax.setAt(.{ batchIdx, outputIndex }, expVal);
                sumExpPerBatch.setAt(.{batchIdx}, sumExpPerBatch.at(.{batchIdx}) + expVal);
            }
        }

        // Step 3: Normalize to get probabilities
        for (0..batchSize) |batchIdx| {
            for (0..self.outputChannel) |outputIndex| {
                softmax.setAt(.{ batchIdx, outputIndex }, softmax.at(.{ batchIdx, outputIndex }) / sumExpPerBatch.at(.{batchIdx}));
            }
        }

        return softmax;
    }

    pub fn backPropagation(self: *Self, dL_dZ: NDArray(f64, 2), X: NDArray(f64, 2), learningRate: f64, allocator: Allocator) !NDArray(f64, 2) {
        // lossGradientForOutput: [batchSize, outputChannel]
        // lastInput: [batchSize, inputChannel]

        const Error = error{ LossGradientForOutputAllZero, EmptyBatch };

        //   Performs a backward pass of the softmax layer.
        //   Returns the loss gradient for this layer's inputs.
        //   - d_L_d_out is the loss gradient for this layer's outputs.
        //   - learn_rate is a float.
        const batchSize = X.shape[0];
        if (batchSize == 0) {
            return Error.EmptyBatch;
        }

        // # Gradients of loss against weights/biases/input
        // d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
        // increase the dimension. 1d @ 1d -> 2d
        var dL_dW = try NDArray(f64, 2).init(.{ self.inputChannel, self.outputChannel }, allocator);
        defer dL_dW.deinit();

        for (0..self.inputChannel) |inputIndex| {
            for (0..self.outputChannel) |outputIndex| {
                var sum: f64 = 0.0;
                for (0..batchSize) |batchIndex| {
                    sum += X.atConst(.{ batchIndex, inputIndex }) * dL_dZ.atConst(.{ batchIndex, outputIndex });
                }
                dL_dW.setAt(.{ inputIndex, outputIndex }, sum);
            }
        }

        var dL_dB = try NDArray(f64, 1).init(.{self.outputChannel}, allocator);
        defer dL_dB.deinit();
        for (0..self.outputChannel) |outputIndex| {
            var sum: f64 = 0.0;
            for (0..batchSize) |batchIndex| {
                sum += dL_dZ.atConst(.{ batchIndex, outputIndex });
            }
            dL_dB.setAt(.{outputIndex}, sum);
        }

        var dL_dX = try NDArray(f64, 2).init(.{ batchSize, self.inputChannel }, allocator);

        for (0..batchSize) |batchIndex| {
            for (0..self.inputChannel) |inputIndex| {
                var sum: f64 = 0.0;
                for (0..self.outputChannel) |outputIndex| {
                    sum += self.weights.at(.{ inputIndex, outputIndex }) * dL_dZ.atConst(.{ batchIndex, outputIndex });
                }
                dL_dX.setAt(.{ batchIndex, inputIndex }, sum);
            }
        }

        // Update weights / biases
        dL_dW.multiplyScalarMut(-1 * learningRate / @as(f64, @floatFromInt(batchSize)));
        try self.weights.addMatrixMut(dL_dW);

        dL_dB.multiplyScalarMut(-1 * learningRate / @as(f64, @floatFromInt(batchSize)));
        try self.bias.addMatrixMut(dL_dB);

        return dL_dX;
    }
};
