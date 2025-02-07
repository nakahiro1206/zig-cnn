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

        var prng = RndGen.init(0);
        const rand = prng.random();

        for (0..inputChannel) |i| {
            for (0..outputChannel) |o| {
                weights.setAt(.{ i, o }, rand.float(f64) / @as(f64, @floatFromInt(inputChannel)));
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

    fn setExpTotalAndReturnSum(self: *Self, input: NDArray(f64, 2), arr: NDArray(f64, 2), allocator: std.mem.Allocator) NDArray(f64, 1) {
        // input: [batchSize, inputChannel]
        // arr: [batchSize, outputChannel]
        const batchSize = input.shape[0];
        var expSum = try NDArray(f64, 1).initWithValue(.{batchSize}, 0.0, allocator);
        for (0..batchSize) |batchIdx| {
            // var expSum: f64 = 0.0;
            for (0..self.outputChannel) |outputIndex| {
                var total: f64 = 0.0;
                for (0..self.inputChannel) |inputChannelIndex| {
                    total += input.at(.{ batchIdx, inputChannelIndex }) * self.weights.at(.{ inputChannelIndex, outputIndex });
                }
                total += self.bias.at(.{outputIndex});
                const expTotal: f64 = @exp(total);
                arr.setAt(.{ batchIdx, outputIndex }, expTotal);
                expSum.setAt(.{batchIdx}, expTotal + expSum.at(.{batchIdx}));
            }
        }
        return expSum;
    }

    pub fn forward(self: *Self, input: NDArray(f64, 2), allocator: Allocator) !NDArray(f64, 2) {
        //   Performs a forward pass of the softmax layer using the given input.
        //   Returns a 1d numpy array containing the respective probability values.
        //   - input can be any array with any dimensions.
        const batchSize = input.shape[0];
        var softmax = try NDArray(f64, 2).initWithValue(.{ batchSize, self.outputChannel }, 0.0, allocator);

        // const setAtFn = softmax.setAt;
        const expSum = try self.setExpTotalAndReturnSum(input, softmax, allocator);
        defer expSum.deinit();

        // if (expSum == 0.0) {
        //     return error.DivisionByZero;
        // }

        for (0..batchSize) |batchIdx| {
            for (0..self.outputChannel) |outputIndex| {
                softmax.setAt(.{ batchIdx, outputIndex }, softmax.at(.{ batchIdx, outputIndex }) / expSum.at(.{batchIdx}));
            }
        }

        return softmax;
    }

    pub fn backPropagation(self: *Self, lossGradientForOutput: NDArray(f64, 2), lastInput: NDArray(f64, 2), learningRate: f64, allocator: Allocator) !NDArray(f64, 1) {
        // lossGradientForOutput: [batchSize, outputChannel]
        // lastInput: [batchSize, inputChannel]

        const Error = error{ LossGradientForOutputAllZero, EmptyBatch };

        //   Performs a backward pass of the softmax layer.
        //   Returns the loss gradient for this layer's inputs.
        //   - d_L_d_out is the loss gradient for this layer's outputs.
        //   - learn_rate is a float.
        const batchSize = lastInput.shape[0];
        if (batchSize == 0) {
            return Error.EmptyBatch;
        }
        //////////////// FROM HERE
        var expTotal = try NDArray(f64, 2).init(.{ batchSize, self.outputChannel }, allocator);
        defer expTotal.deinit();

        const expSum = self.setExpTotalAndReturnSum(lastInput, expTotal);

        var OutputGradientForTotals = try NDArray(f64, 2).init(.{ batchSize, self.outputChannel }, allocator);
        defer OutputGradientForTotals.deinit();

        var groundTruthIndex: usize = 0;
        for (lossGradientForOutput.items, 0..) |gradient, index| {
            if (gradient == 0.0) {
                continue;
            }
            groundTruthIndex = index;
            break;
        }
        for (0..batchSize) |batchIndex| {
            for (0..self.outputChannel) |outputIndex| {
                const expOutput = expTotal.at(.{ batchIndex, outputIndex });

                const expSumOfBatch: f64 = expSum.at(.{batchIndex});

                if (outputIndex == groundTruthIndex) {
                    // d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
                    OutputGradientForTotals.setAt(.{ batchIndex, outputIndex }, expOutput * (expSumOfBatch - expOutput) / (expSumOfBatch * expSumOfBatch));
                } else {
                    // d_out_d_t[j!=i] = -t_exp[i] * t_exp / (S ** 2)
                    OutputGradientForTotals.setAt(.{ batchIndex, outputIndex }, (-1) * expOutput * expTotal.at(.{ batchIndex, outputIndex }) / (expSumOfBatch * expSumOfBatch));
                }
            }
        }

        // # Gradients of loss against totals
        // d_L_d_t = gradient * d_out_d_t
        var lossGradientForTotals = try NDArray(f64, 2).duplicate(.{ batchSize, self.outputChannel }, OutputGradientForTotals, allocator);
        defer lossGradientForTotals.deinit();

        lossGradientForTotals.multiplyElementwiseMut(lossGradientForOutput);

        // # Gradients of loss against weights/biases/input
        // d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
        // increase the dimension. 1d @ 1d -> 2d
        var lossGradientForWeights = try NDArray(f64, 2).init(.{ self.inputChannel, self.outputChannel }, allocator);
        defer lossGradientForWeights.deinit();

        for (0..self.inputChannel) |inputIndex| {
            for (0..self.outputChannel) |outputIndex| {
                var sum: f64 = 0.0;
                for (0..batchSize) |batchIndex| {
                    sum += lastInput.at(.{ batchIndex, inputIndex }) * lossGradientForTotals.at(.{ batchIndex, outputIndex });
                }
                lossGradientForWeights.setAt(.{ inputIndex, outputIndex }, sum / @as(f64, @floatFromInt(batchSize)));
            }
        }

        //     d_L_d_b = d_L_d_t * d_t_d_b
        var lossGradientForBias = try NDArray(f64, 1).initWithValue(.{self.outputChannel}, 0.0, allocator);
        defer lossGradientForBias.deinit();

        for (0..self.outputChannel) |outputIndex| {
            var sum: f64 = 0.0;
            for (0..batchSize) |batchIndex| {
                sum += lossGradientForTotals.at(.{ batchIndex, outputIndex });
            }
            lossGradientForBias.setAt(.{outputIndex}, sum / @as(f64, @floatFromInt(batchSize)));
        }

        //     d_L_d_inputs = d_t_d_inputs @ d_L_d_t
        var lossGradientForInputs = try NDArray(f64, 2).init(.{ batchSize, self.inputChannel }, allocator);
        defer lossGradientForInputs.deinit();

        for (0..batchSize) |batchIndex| {
            for (0..self.inputChannel) |inputIndex| {
                var sum: f64 = 0.0;
                for (0..self.outputChannel) |outputIndex| {
                    sum += self.weights.at(.{ inputIndex, outputIndex }) * lossGradientForTotals.at(.{ batchIndex, outputIndex });
                }
                lossGradientForInputs.setAt(.{ batchIndex, inputIndex }, sum);
            }
        }

        // # Update weights / biases
        //  //     self.weights -= learn_rate * d_L_d_w
        lossGradientForWeights.multiplyScalarMut(-1 * learningRate);
        try self.weights.addMatrixMut(lossGradientForWeights);
        //     self.biases -= learn_rate * d_L_d_b
        lossGradientForBias.multiplyScalarMut(-1 * learningRate);
        try self.bias.addMatrixMut(lossGradientForBias);
        //     return d_L_d_inputs.reshape(self.last_input_shape)
        return lossGradientForInputs;
        // ///////////////////////////
        // for (lossGradientForOutput.items, 0..) |gradient, index| {
        //     if (gradient == 0.0) {
        //         continue;
        //     }
        //     var expTotal = try NDArray(f64, 1).initWithValue(.{self.outputChannel}, 0.0, allocator);
        //     defer expTotal.deinit();
        //     const expSum = self.setExpTotalAndReturnSum(lastInput, expTotal);

        //     // if (expSum == 0.0) {
        //     //     return error.DivisionByZero;
        //     // }

        //     // Here, index == answer label's index.
        //     var OutputGradientForTotals = try NDArray(f64, 2).init(.{ batchSize, self.outputChannel }, allocator);
        //     defer OutputGradientForTotals.deinit();

        //     const expOutput = expTotal.at(.{index});
        //     for (0..self.outputChannel) |outputIndex| {
        //         if (outputIndex == index) {
        //             // d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
        //             OutputGradientForTotals.setAt(.{outputIndex}, expOutput * (expSum - expOutput) / (expSum * expSum));
        //         } else {
        //             // d_out_d_t[j!=i] = -t_exp[i] * t_exp / (S ** 2)
        //             OutputGradientForTotals.setAt(.{outputIndex}, (-1) * expOutput * expTotal.at(.{outputIndex}) / (expSum * expSum));
        //         }
        //     }

            // // # Gradients of totals against weights/biases/input
            // const totalsGradientForWeights = lastInput;
            // const totalsGradientForBias: f64 = 1.0;
            // const totalsGradientForInputs = self.weights;

            // // # Gradients of loss against totals
            // // d_L_d_t = gradient * d_out_d_t
            // var lossGradientForTotals = try NDArray(f64, 1).duplicate(.{self.outputChannel}, OutputGradientForTotals, allocator);
            // defer lossGradientForTotals.deinit();
            // lossGradientForTotals.multiplyScalarMut(gradient);

            // // # Gradients of loss against weights/biases/input
            // // d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            // // increase the dimension. 1d @ 1d -> 2d
            // var lossGradientForWeights = try NDArray(f64, 2).init(.{ lastInput.shape[0], self.outputChannel }, allocator);
            // defer lossGradientForWeights.deinit();
            // for (0..lastInput.shape[0]) |lastInputIndex| {
            //     for (0..self.outputChannel) |outputChannelIndex| {
            //         const val = totalsGradientForWeights.atConst(.{lastInputIndex}) * lossGradientForTotals.at(.{outputChannelIndex});
            //         lossGradientForWeights.setAt(.{ lastInputIndex, outputChannelIndex }, val);
            //     }
            // }

            // //     d_L_d_b = d_L_d_t * d_t_d_b
            // var lossGradientForBias = try NDArray(f64, 1).fromCopiedSlice(.{self.outputChannel}, lossGradientForTotals.items, allocator);
            // defer lossGradientForBias.deinit();
            // lossGradientForBias.multiplyScalarMut(totalsGradientForBias);

            //     d_L_d_inputs = d_t_d_inputs @ d_L_d_t
            var lossGradientForInputs = try NDArray(f64, 1).init(lastInput.shape, allocator);
            for (0..lastInput.shape[0]) |lastInputIndex| {
                var sum: f64 = 0.0;
                for (0..self.outputChannel) |outputChannelIndex| {
                    const val = totalsGradientForInputs.atConst(.{ lastInputIndex, outputChannelIndex }) * lossGradientForTotals.at(.{outputChannelIndex});
                    sum += val;
                }
                lossGradientForInputs.setAt(.{lastInputIndex}, sum);
            }

            //     # Update weights / biases
            //     self.weights -= learn_rate * d_L_d_w
            lossGradientForWeights.multiplyScalarMut(-1 * learningRate);
            try self.weights.addMatrixMut(lossGradientForWeights);
            //     self.biases -= learn_rate * d_L_d_b
            lossGradientForBias.multiplyScalarMut(-1 * learningRate);
            try self.bias.addMatrixMut(lossGradientForBias);

            //     return d_L_d_inputs.reshape(self.last_input_shape)
            return lossGradientForInputs;
        }

        return Error.LossGradientForOutputAllZero;
    }
};
