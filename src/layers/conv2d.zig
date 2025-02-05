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

        var prng = RndGen.init(0);
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
        // self.lastInput.deinit();
    }

    pub fn print(self: *Self) void {
        std.debug.print("Conv2DLayer\n", .{});
        std.debug.print("filterShape: {}\n", .{self.filterShape});
        std.debug.print("filters: {}\n", .{self.filters});
    }

    pub fn iterateRegions(self: *Self, image: NDArray(f64, 2)) void {
        // Generates all possible filterHeight x filterWidth image regions using valid padding.
        // - image is a 2d numpy array.

        // 0: outputChannel, 1: inputChannel, 2: filterHeight, 3: filterWidth
        const filterHeight = self.filterShape[2];
        const filterWidth = self.filterShape[3];
        const imageHeight = image.shape[0];
        const imageWidth = image.shape[1];

        for (0..imageHeight - filterHeight + 1) |i| {
            for (0..imageWidth - filterWidth + 1) |j| {
                const region = image.slice(.{ i, j }, .{ filterHeight, filterWidth });
                self.lastInput = region;
            }
        }
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
                            output.setAt(.{ batchIndex, outputChannelIdx, inputH, inputW }, sum);
                        }
                    }
                }
            }
        }

        return output;
    }

    pub fn backPropagation(self: *Self, lossGradientForOutput: NDArray(f64, 4), inputUsedAtForward: NDArray(f64, 4), learningRate: f64, allocator: std.mem.Allocator) !NDArray(f64, 4) {
        // Performs a backward pass of the conv layer.
        // - d_L_d_out is the loss gradient for this layer's outputs. size: (filter_num, 28 - 2, 28 - 2`)
        // - learn_rate is a float.

        const batchSize = inputUsedAtForward.shape[0];
        const lastInputChannel = inputUsedAtForward.shape[1];
        const lastInputHeight = inputUsedAtForward.shape[2];
        const lastInputWidth = inputUsedAtForward.shape[3];

        // const inputChannel = self.filterShape[1]; == lastInputChannel
        const outputChannel = self.filterShape[0];
        const filterHeight = self.filterShape[2];
        const filterWidth = self.filterShape[3];

        var lossGradientForFilter = try NDArray(f64, 4).init(.{ batchSize, lastInputChannel, lastInputHeight, lastInputWidth }, allocator);

        for (0..batchSize) |batchIdx| {
            for (0..lastInputChannel) |inputChannelIdx| {
                for (0..lastInputHeight - filterHeight + 1) |lastInputH| {
                    for (0..lastInputWidth - filterWidth + 1) |lastInputW| {
                        var sum: f64 = 0.0;
                        for (0..filterHeight) |filterH| {
                            for (0..filterWidth) |filterW| {
                                for (0..outputChannel) |outputChannelIdx| {
                                    const lossGradVal = lossGradientForOutput.atConst(.{ batchIdx, outputChannelIdx, lastInputH, lastInputW });
                                    const inputVal = inputUsedAtForward.atConst(.{ batchIdx, inputChannelIdx, lastInputH + filterH, lastInputW + filterW });
                                    sum += lossGradVal * inputVal;
                                }
                            }
                        }
                        // Clamp the sum to a reasonable range to prevent overflow/underflow
                        if (sum > 1e10) sum = 1e10;
                        if (sum < -1e10) sum = -1e10;
                        lossGradientForFilter.setAt(.{ batchIdx, inputChannelIdx, lastInputH, lastInputW }, sum);
                    }
                }
            }
        }

        // update filters.
        for (0..outputChannel) |outputChannelIdx| {
            for (0..lastInputChannel) |inputChannelIdx| {
                for (0..filterHeight) |filterH| {
                    for (0..filterWidth) |filterW| {
                        const tmp: f64 = self.filters.at(.{ outputChannelIdx, inputChannelIdx, filterH, filterW });
                        var diff: f64 = 0.0;
                        for (0..batchSize) |batchIdx| {
                            diff += lossGradientForFilter.atConst(.{ batchIdx, inputChannelIdx, filterH, filterW });
                        }
                        var newVal = tmp - diff * learningRate;
                        // Clamp the new value to a reasonable range to prevent overflow/underflow
                        if (newVal > 1e10) newVal = 1e10;
                        if (newVal < -1e10) newVal = -1e10;
                        self.filters.setAt(.{ outputChannelIdx, inputChannelIdx, filterH, filterW }, newVal);
                    }
                }
            }
        }

        // update filter bias
        //
        return lossGradientForFilter;
    }
};
