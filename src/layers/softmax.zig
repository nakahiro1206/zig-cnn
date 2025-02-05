const std = @import("std");
const Allocator = std.mem.Allocator;
const NDArray = @import("../pblischak/zig-ndarray/ndarray.zig").NDArray;
const RndGen = std.rand.DefaultPrng;

pub const Softmax = struct {
    inputSize: [1]usize, // [filters, height, width]: example [3 * 3 * 3]
    weights: NDArray(f64, 2),
    bias: NDArray(f64, 1),
    outputSize: usize,

    const Self = @This();

    pub fn init(inputSize: [1]usize, outputSize: usize, allocator: std.mem.Allocator) !Self {
        const flattenSize = inputSize[0];
        var weights: NDArray(f64, 2) = try NDArray(f64, 2).init(.{ flattenSize, outputSize }, allocator);
        const bias: NDArray(f64, 1) = try NDArray(f64, 1).initWithValue(.{outputSize}, 0, allocator);

        var prng = RndGen.init(0);
        const rand = prng.random();

        for (0..flattenSize) |h| {
            for (0..outputSize) |w| {
                weights.setAt(.{ h, w }, rand.float(f64) / @as(f64, @floatFromInt(flattenSize)));
            }
        }

        return Self{
            .inputSize = inputSize,
            .weights = weights,
            .bias = bias,
            .outputSize = outputSize,
        };
    }

    pub fn dealloc(self: *Self) void {
        self.weights.deinit();
        self.bias.deinit();
    }

    fn setExpTotalAndReturnSum(self: *Self, input: NDArray(f64, 1), arr: NDArray(f64, 1)) f64 {
        var expSum: f64 = 0.0;
        for (0..self.outputSize) |outputIndex| {
            var total: f64 = 0.0;
            for (input.items, 0..) |item, index| {
                total += item * self.weights.at(.{ index, outputIndex });
            }
            total += self.bias.at(.{outputIndex});
            const expTotal: f64 = @exp(total);
            arr.setAtConst(.{outputIndex}, expTotal);
            expSum += expTotal;
        }
        return expSum;
    }

    pub fn forward(self: *Self, input: NDArray(f64, 1), allocator: Allocator) !NDArray(f64, 1) {
        //   Performs a forward pass of the softmax layer using the given input.
        //   Returns a 1d numpy array containing the respective probability values.
        //   - input can be any array with any dimensions.
        var softmax = try NDArray(f64, 1).initWithValue(.{self.outputSize}, 0.0, allocator);

        // const setAtFn = softmax.setAt;
        const expSum = self.setExpTotalAndReturnSum(input, softmax);

        if (expSum == 0.0) {
            return error.DivisionByZero;
        }

        for (0..self.outputSize) |outputIndex| {
            softmax.setAt(.{outputIndex}, softmax.at(.{outputIndex}) / expSum);
        }

        return softmax;
    }

    pub fn backPropagation(self: *Self, lossGradientForOutput: NDArray(f64, 1), lastInput: NDArray(f64, 1), learningRate: f64, allocator: Allocator) !NDArray(f64, 1) {
        const Error = error{
            LossGradientForOutputAllZero,
        };

        //   Performs a backward pass of the softmax layer.
        //   Returns the loss gradient for this layer's inputs.
        //   - d_L_d_out is the loss gradient for this layer's outputs.
        //   - learn_rate is a float.
        for (lossGradientForOutput.items, 0..) |gradient, index| {
            if (gradient == 0.0) {
                continue;
            }
            var expTotal = try NDArray(f64, 1).initWithValue(.{self.outputSize}, 0.0, allocator);
            defer expTotal.deinit();
            const expSum = self.setExpTotalAndReturnSum(lastInput, expTotal);

            if (expSum == 0.0) {
                return error.DivisionByZero;
            }

            // Here, index == answer label's index.
            var OutputGradientForTotals = try NDArray(f64, 1).init(.{self.outputSize}, allocator);
            defer OutputGradientForTotals.deinit();

            const expOutput = expTotal.at(.{index});
            for (0..self.outputSize) |outputIndex| {
                if (outputIndex == index) {
                    // d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
                    OutputGradientForTotals.setAt(.{outputIndex}, expOutput * (expSum - expOutput) / (expSum * expSum));
                } else {
                    // d_out_d_t[j!=i] = -t_exp[i] * t_exp / (S ** 2)
                    OutputGradientForTotals.setAt(.{outputIndex}, (-1) * expOutput * expTotal.at(.{outputIndex}) / (expSum * expSum));
                }
            }

            // # Gradients of totals against weights/biases/input
            const totalsGradientForWeights = lastInput;
            const totalsGradientForBias: f64 = 1.0;
            const totalsGradientForInputs = self.weights;

            // # Gradients of loss against totals
            // d_L_d_t = gradient * d_out_d_t
            var lossGradientForTotals = try NDArray(f64, 1).duplicate(.{self.outputSize}, OutputGradientForTotals, allocator);
            defer lossGradientForTotals.deinit();
            lossGradientForTotals.multiplyScalarMut(gradient);

            // # Gradients of loss against weights/biases/input
            // d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            // increase the dimension. 1d @ 1d -> 2d
            var lossGradientForWeights = try NDArray(f64, 2).init(.{ lastInput.shape[0], self.outputSize }, allocator);
            defer lossGradientForWeights.deinit();
            for (0..lastInput.shape[0]) |lastInputIndex| {
                for (0..self.outputSize) |outputSizeIndex| {
                    const val = totalsGradientForWeights.atConst(.{lastInputIndex}) * lossGradientForTotals.at(.{outputSizeIndex});
                    lossGradientForWeights.setAt(.{ lastInputIndex, outputSizeIndex }, val);
                }
            }

            //     d_L_d_b = d_L_d_t * d_t_d_b
            var lossGradientForBias = try NDArray(f64, 1).fromCopiedSlice(.{self.outputSize}, lossGradientForTotals.items, allocator);
            defer lossGradientForBias.deinit();
            lossGradientForBias.multiplyScalarMut(totalsGradientForBias);

            //     d_L_d_inputs = d_t_d_inputs @ d_L_d_t
            var lossGradientForInputs = try NDArray(f64, 1).init(lastInput.shape, allocator);
            for (0..lastInput.shape[0]) |lastInputIndex| {
                var sum: f64 = 0.0;
                for (0..self.outputSize) |outputSizeIndex| {
                    const val = totalsGradientForInputs.atConst(.{ lastInputIndex, outputSizeIndex }) * lossGradientForTotals.at(.{outputSizeIndex});
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
