// pub const Conv2DLayer = struct {
//     filters: [][][f32], // 3D kernels
//     bias: []f32,
//     activation: fn (f32) f32, // Pointer to activation function.
// };

// fn conv2d(input: Tensor3D, layer: Conv2DLayer) Tensor3D {
//     // Sliding window implementation.
//     // apply relu activation function.
// }
//

const std = @import("std");
const NDArray = @import("../pblischak/zig-ndarray/ndarray.zig").NDArray;
const RndGen = std.rand.DefaultPrng;

pub const Conv2DLayer = struct {
    filters_num: usize,
    filter_size: [2]usize, // [height, width]: example [3, 3]
    filters: NDArray(f64, 3), // Filters with dimensions (num_filters, 3, 3)
    lastInput: NDArray(f64, 2), // Stores the last input for backpropagation

    const Self = @This();

    pub fn init(filters_num: usize, filter_size: [2]usize, allocator: std.mem.Allocator) !Self {
        var filters: NDArray(f64, 3) = try NDArray(f64, 3).init(.{ filters_num, filter_size[0], filter_size[1] }, allocator);

        var prng = RndGen.init(0);
        const rand = prng.random();

        for (0..filters_num) |i| {
            for (0..filter_size[0]) |j| {
                for (0..filter_size[1]) |k| {
                    filters.setAt(.{ i, j, k }, rand.float(f64));
                }
            }
        }

        const lastInput: NDArray(f64, 2) = try NDArray(f64, 2).init(.{ 10, 10 }, allocator);

        return Self{
            .filters_num = filters_num,
            .filter_size = filter_size,
            .filters = filters,
            .lastInput = lastInput,
        };
    }

    pub fn dealloc(self: *Self) void {
        self.filters.deinit();
        // self.lastInput.deinit();
    }

    pub fn print(self: *Self) void {
        std.debug.print("Conv2DLayer\n", .{});
        std.debug.print("filters_num: {}\n", .{self.filters_num});
        std.debug.print("filter_size: [{}, {}]\n", .{ self.filter_size[0], self.filter_size[1] });
        std.debug.print("filters: {}\n", .{self.filters});
    }

    pub fn iterateRegions(self: *Self, image: NDArray(f64, 2)) void {
        // Generates all possible filterHeight x filterWidth image regions using valid padding.
        // - image is a 2d numpy array.
        const filterHeight = self.filter_size[0];
        const filterWidth = self.filter_size[1];
        const imageHeight = image.shape[0];
        const imageWidth = image.shape[1];

        for (0..imageHeight - filterHeight + 1) |i| {
            for (0..imageWidth - filterWidth + 1) |j| {
                const region = image.slice(.{ i, j }, .{ filterHeight, filterWidth });
                self.lastInput = region;
            }
        }
    }

    pub fn forward(self: *Self, input: NDArray(f64, 2), allocator: std.mem.Allocator) !NDArray(f64, 3) {
        // Performs a forward pass of the conv layer using the given input.
        //     Returns a 3d numpy array with dimensions (h, w, num_filters).
        // - input is a 2d numpy array

        self.lastInput.deinit();
        self.lastInput = input;

        const filterHeight = self.filter_size[0];
        const filterWidth = self.filter_size[1];
        const inputHeight = input.shape[0];
        const inputWidth = input.shape[1];

        var output = try NDArray(f64, 3).initWithValue(
            .{ inputHeight - filterHeight + 1, inputWidth - filterWidth + 1, self.filters_num },
            0.0,
            allocator,
        );

        for (0..self.filters_num) |filterIdx| {
            for (0..inputHeight - filterHeight + 1) |inputH| {
                for (0..inputWidth - filterWidth + 1) |inputW| {
                    var sum: f64 = 0.0;
                    // output[idx, h, w] += input[h:h+filterHeight, w:w+filterWidth] * self.filters[idx]
                    for (0..filterHeight) |filterH| {
                        for (0..filterWidth) |filterW| {
                            sum += input.atConst(.{ inputH + filterH, inputW + filterW }) * self.filters.at(.{ filterIdx, filterH, filterW });
                        }
                    }
                    output.setAt(.{ filterIdx, inputH, inputW }, sum);
                }
            }
        }

        return output;
    }

    pub fn backPropagation(self: *Self, lossGradientForOutput: NDArray(f64, 3), learningRate: f64, allocator: std.mem.Allocator) !void {
        // Performs a backward pass of the conv layer.
        // - d_L_d_out is the loss gradient for this layer's outputs. size: (filter_num, 28 - 2, 28 - 2`)
        // - learn_rate is a float.
        const lastInputHeight = self.lastInput.shape[0];
        const lastInputWidth = self.lastInput.shape[1];
        const filterHeight = self.filter_size[0];
        const filterWidth = self.filter_size[1];

        var lossGradientForFilter = try NDArray(f64, 3).init(.{ self.filters_num, filterHeight, filterWidth }, allocator);
        defer lossGradientForFilter.deinit();

        for (0..self.filters_num) |filterIdx| {
            // lossGradientForFilter[idx] += sum(lossGradientForOutput[idx, h, w] *
            // lastInput[h:h+filterHeight, w:w+filterWidth])
            for (0..filterHeight) |filterH| {
                for (0..filterWidth) |filterW| {
                    var sum: f64 = 0.0;
                    for (0..lastInputHeight - filterHeight + 1) |lastInputH| {
                        for (0..lastInputWidth - filterWidth + 1) |lastInputW| {
                            sum += lossGradientForOutput.atConst(.{ filterIdx, lastInputH, lastInputW }) * self.lastInput.atConst(.{ lastInputH + filterH, lastInputW + filterW });
                        }
                    }
                    lossGradientForFilter.setAt(.{ filterIdx, filterH, filterW }, sum);
                }
            }
        }

        // update filters.
        for (0..self.filters_num) |filterIdx| {
            for (0..filterHeight) |filterH| {
                for (0..filterWidth) |filterW| {
                    const tmp: f64 = self.filters.at(.{ filterIdx, filterH, filterW });
                    const diff: f64 = learningRate * lossGradientForFilter.at(.{ filterIdx, filterH, filterW });
                    self.filters.setAt(.{ filterIdx, filterH, filterW }, tmp - diff);
                }
            }
        }
    }

    // /// Iterates over all possible 3x3 regions in the input image.
    //     fn iterateRegions(image: [][]f64) []const struct {
    //         im_region: [3][3]f64,
    //         i: usize,
    //         j: usize,
    //     } {
    //         const h = image.len;
    //         const w = if (h > 0) image[0].len else 0;
    //         var regions: []const struct {
    //             im_region: [3][3]f64,
    //             i: usize,
    //             j: usize,
    //         } = &[]; // Create a container for regions (dynamic array)

    //         for (0..h - 2) |j| {
    //             region
    //         consts and ensures output
    // }
    // pub fn alloc(allocator: std.mem.Allocator) !Self {
    //     var w = linalg.Matrix.new(OUT, IN);
    //     try w.alloc(allocator);
    //     var b = linalg.Matrix.new(OUT, 1);
    //     try b.alloc(allocator);
    //     return Self{ .weights = w, .biases = b };
    // }

    // pub fn dealloc(self: Self, allocator: std.mem.Allocator) void {
    //     self.biases.dealloc(allocator);
    //     self.weights.dealloc(allocator);
    // }
};
