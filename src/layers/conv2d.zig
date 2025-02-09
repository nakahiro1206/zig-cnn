const std = @import("std");
const NDArray = @import("../pblischak/zig-ndarray/ndarray.zig").NDArray;
const RndGen = std.rand.DefaultPrng;

pub const Conv2DLayer = struct {
    filterShape: [4]usize, // [outputChannel, inputChannel, height, width]: example [32, 1, 3, 3]
    filters: NDArray(f64, 4), // Filters with dimensions like (32, 1, 3, 3)
    filterBias: NDArray(f64, 1), // 1D array to store filter bias sized [outputChannel].

    const Self = @This();

    pub fn init(filterShape: [4]usize, allocator: std.mem.Allocator) !Self {
        var filters: NDArray(f64, 4) = try NDArray(f64, 4).init(filterShape, allocator);
        var filterBias: NDArray(f64, 1) = try NDArray(f64, 1).init(.{filterShape[0]}, allocator);

        var prng = RndGen.init(@as(u64, @intCast(std.time.milliTimestamp())));
        const rand = prng.random();

        for (0..filterShape[0]) |i| {
            // set random value for filters
            for (0..filterShape[1]) |j| {
                for (0..filterShape[2]) |k| {
                    for (0..filterShape[3]) |l| {
                        filters.setAt(.{ i, j, k, l }, rand.float(f64));
                    }
                }
            }

            // set random value for filter bias
            filterBias.setAt(.{i}, rand.float(f64));
        }

        return Self{
            .filterShape = filterShape,
            .filters = filters,
            .filterBias = filterBias,
        };
    }

    pub fn dealloc(self: *Self) void {
        self.filters.deinit();
        self.filterBias.deinit();
    }

    pub fn print(self: *Self) void {
        std.debug.print("Conv2DLayer\n", .{});
        std.debug.print("filterShape: {}\n", .{self.filterShape});
        std.debug.print("filters: {}\n", .{self.filters});
    }

    pub fn forward(self: *Self, input: NDArray(f64, 4), allocator: std.mem.Allocator) !NDArray(f64, 4) {
        // Performs a forward pass of the conv layer using the given input.
        //     Returns a 3d numpy array with dimensions (h, w, num_filters).
        // - input is a 2d numpy array

        // 0: outputChannel, 1: inputChannel, 2: filterHeight, 3: filterWidth
        const outputChannel = self.filterShape[0];
        const inputChannel = self.filterShape[1];
        const filterHeight = self.filterShape[2];
        const filterWidth = self.filterShape[3];

        const batchSize = input.shape[0];
        // const inputChannel = input.shape[1];
        const inputHeight = input.shape[2];
        const inputWidth = input.shape[3];

        var output = try NDArray(f64, 4).initWithValue(
            .{ batchSize, outputChannel, inputHeight - filterHeight + 1, inputWidth - filterWidth + 1 },
            0.0,
            allocator,
        );

        for (0..batchSize) |batchIndex| {
            for (0..outputChannel) |outputChannelIdx| {
                for (0..inputChannel) |inputChannelIdx| {
                    for (0..inputHeight - filterHeight + 1) |inputH| {
                        for (0..inputWidth - filterWidth + 1) |inputW| {
                            var sum: f64 = 0.0;
                            // output[idx, h, w] += input[h:h+filterHeight, w:w+filterWidth] * self.filters[idx]
                            for (0..filterHeight) |filterH| {
                                for (0..filterWidth) |filterW| {
                                    const inputVal = input.atConst(.{ batchIndex, inputChannelIdx, inputH + filterH, inputW + filterW });
                                    const filterVal = self.filters.at(.{ outputChannelIdx, inputChannelIdx, filterH, filterW });
                                    sum += inputVal * filterVal;
                                }
                            }
                            // Clamp the sum to a reasonable range to prevent overflow/underflow
                            if (sum > 1e10) sum = 1e10;
                            if (sum < -1e10) sum = -1e10;
                            sum += self.filterBias.at(.{outputChannelIdx}); // Add bias term
                            output.setAt(.{ batchIndex, outputChannelIdx, inputH, inputW }, sum);
                        }
                    }
                }
            }
        }

        return output;
    }

    pub fn backPropagation(self: *Self, dL_dZ: NDArray(f64, 4), X: NDArray(f64, 4), learningRate: f64, allocator: std.mem.Allocator) !NDArray(f64, 4) {
        // Performs a backward pass of the conv layer.
        // - d_L_d_out is the loss gradient for this layer's outputs. size: (filter_num, 28 - 2, 28 - 2`)
        // - learn_rate is a float.

        const B = X.shape[0];
        const C_X = X.shape[1];
        const H_X = X.shape[2];
        const W_X = X.shape[3];

        // const inputChannel = self.filterShape[1]; == lastInputChannel
        const C_Z = dL_dZ.shape[1];
        const H_Z = self.filterShape[2];
        const W_Z = self.filterShape[3];

        const H_F = self.filterShape[2];
        const W_F = self.filterShape[3];

        var dL_dX = try NDArray(f64, 4).init(.{ B, C_X, H_X, W_X }, allocator);

        for (0..B) |b| {
            for (0..C_X) |c_X| {
                for (0..H_X) |h_X| {
                    for (0..W_X) |w_X| {
                        var sum: f64 = 0.0;
                        for (0..C_Z) |c_Z| {
                            for (0..H_F) |h| {
                                for (0..W_F) |w| {
                                    if (h_X < h or w_X < w) {
                                        continue;
                                    }
                                    const h_index = h_X - h;
                                    const w_index = w_X - w;

                                    if (h_index < H_X - H_F + 1 and w_index < W_X - W_F + 1) {
                                        const lossGradVal = dL_dZ.atConst(.{ b, c_Z, h_index, w_index });
                                        const filterVal = self.filters.at(.{ c_Z, c_X, h, w });
                                        sum += lossGradVal * filterVal;
                                    }
                                }
                            }
                        }
                        dL_dX.setAt(.{ b, c_X, h_X, w_X }, sum);
                    }
                }
            }
        }

        // update filters.
        for (0..C_Z) |c_Z| {
            for (0..C_X) |c_X| {
                for (0..H_F) |h_F| {
                    for (0..W_F) |w_F| {
                        var filterGradSum: f64 = 0.0;
                        // dL_dW = sum_{b, h, w} dL_dZ[b, c_F, h, w] * X[b, c_X, h + h_F, w + w_F]
                        for (0..B) |b| {
                            for (0..H_Z) |h_Z| {
                                for (0..W_Z) |w_Z| {
                                    const lossGradVal = dL_dZ.atConst(.{ b, c_Z, h_Z, w_Z });
                                    const inputVal = X.atConst(.{ b, c_X, h_Z + h_F, w_Z + w_F });
                                    filterGradSum += lossGradVal * inputVal;
                                }
                            }
                        }
                        // Apply weight update
                        // newW = W - learningRate * dL_dW
                        const newWeight = self.filters.at(.{ c_Z, c_X, h_F, w_F }) - learningRate * filterGradSum / @as(f64, @floatFromInt(B));
                        self.filters.setAt(.{ c_Z, c_X, h_F, w_F }, newWeight);
                    }
                }
            }
        }

        // calculate update diff and update filter bias.
        for (0..C_Z) |c_Z| {
            var biasGradSum: f64 = 0.0;
            // biasGradSum = sum_{b, h, w} dL_dZ[b, c_F, h, w]
            for (0..B) |b| {
                for (0..H_X - H_F + 1) |h_Z| {
                    for (0..W_X - W_F + 1) |w_Z| {
                        biasGradSum += dL_dZ.atConst(.{ b, c_Z, h_Z, w_Z });
                    }
                }
            }
            // newB = B - learningRate * dL_dB
            // normalize by batch size
            const newBias: f64 = self.filterBias.at(.{c_Z}) - learningRate * biasGradSum / @as(f64, @floatFromInt(B));
            self.filterBias.setAt(.{c_Z}, newBias);
        }

        return dL_dX;
    }
};
