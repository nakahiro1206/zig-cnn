const std = @import("std");
const Allocator = std.mem.Allocator;
const NDArray = @import("../pblischak/zig-ndarray/ndarray.zig").NDArray;
const RndGen = std.rand.DefaultPrng;

pub const MaxPoolingLayer = struct {
    poolSize: [2]usize, // [height, width]: example [2, 2]

    const Self = @This();

    pub fn init(poolSize: [2]usize) Self {
        return Self{
            .poolSize = poolSize,
        };
    }

    pub fn forward(self: *const Self, input: NDArray(f64, 4), allocator: Allocator) !NDArray(f64, 4) {
        // input: Filters with dimensions [batch_size, channels, height, width]
        const batchSize = input.shape[0];
        const channels = input.shape[1];
        const height = input.shape[2];
        const width = input.shape[3];

        const newHeight: usize = (height + 1) / 2; // ceil(height / 2)
        const newWidth: usize = (width + 1) / 2;

        const heightRatio = self.poolSize[0];
        const widthRatio = self.poolSize[1];

        var output = try NDArray(f64, 4).init(.{ batchSize, channels, newHeight, newWidth }, allocator);

        // assume padding == "same" && stride == input.shape
        const heightStride = self.poolSize[0];
        const widthStride = self.poolSize[1];
        for (0..batchSize) |batchIdx| {
            for (0..channels) |channelIdx| {
                for (0..newHeight) |heightIdx| {
                    for (0..newWidth) |widthIdx| {
                        var maxVal = -1 * std.math.inf(f64); // initialize with negative inf.
                        for (0..heightStride) |heightStrideIdx| {
                            for (0..widthStride) |widthStrideIdx| {
                                const h = heightIdx * heightRatio + heightStrideIdx;
                                const w = widthIdx * widthRatio + widthStrideIdx;
                                if (input.isValidIndex(.{ batchIdx, channelIdx, h, w })) {
                                    maxVal = @max(maxVal, input.atConst(.{ batchIdx, channelIdx, h, w }));
                                }
                            }
                        }
                        output.setAt(.{ batchIdx, channelIdx, heightIdx, widthIdx }, maxVal);
                    }
                }
            }
        }
        return output;
    }

    pub fn backPropagation(self: *const Self, lossGradientForOutput: NDArray(f64, 4), inputUsedAtForward: NDArray(f64, 4), allocator: Allocator) !NDArray(f64, 4) {
        const batchSize = lossGradientForOutput.shape[0];
        const channels = lossGradientForOutput.shape[1];
        const height = lossGradientForOutput.shape[2];
        const width = lossGradientForOutput.shape[3];

        const strideHeight = self.poolSize[0];
        const strideWidth = self.poolSize[1];

        // inputUsedAtForward: [batch_size, channels, height, width]
        const newHeight = inputUsedAtForward.shape[2];
        const newWidth = inputUsedAtForward.shape[3];

        // upscale lossGradientForOutput to the size of inputUsedAtForward, lossGradientForInput
        // copy lossGradientForOutput at the same position of the max pixels in the last input
        // if the value of the pixel is not the max, set 0.
        var lossGradientForInput = try NDArray(f64, 4).initWithValue(.{ batchSize, channels, newHeight, newWidth }, 0.0, allocator);

        for (0..batchSize) |batchIdx| {
            for (0..channels) |channelIdx| {
                for (0..height) |heightIdx| {
                    for (0..width) |widthIdx| {
                        var maxValOfLastInput: f64 = -1 * std.math.inf(f64);
                        for (0..strideHeight) |strideHeightIdx| {
                            for (0..strideWidth) |strideWidthIdx| {
                                const h = heightIdx * strideHeight + strideHeightIdx;
                                const w = widthIdx * strideWidth + strideWidthIdx;
                                if (inputUsedAtForward.isValidIndex(.{ batchIdx, channelIdx, h, w })) {
                                    const valOfLastInput = inputUsedAtForward.atConst(.{ batchIdx, channelIdx, h, w });
                                    if (valOfLastInput > maxValOfLastInput) {
                                        maxValOfLastInput = valOfLastInput;
                                    }
                                }
                            }
                        }
                        for (0..strideHeight) |strideHeightIdx| {
                            for (0..strideWidth) |strideWidthIdx| {
                                const h = heightIdx * strideHeight + strideHeightIdx;
                                const w = widthIdx * strideWidth + strideWidthIdx;
                                if (inputUsedAtForward.isValidIndex(.{ batchIdx, channelIdx, h, w })) {
                                    const valOfLastInput = inputUsedAtForward.atConst(.{ batchIdx, channelIdx, h, w });
                                    if (valOfLastInput == maxValOfLastInput) {
                                        lossGradientForInput.setAt(.{ batchIdx, channelIdx, h, w }, lossGradientForOutput.atConst(.{ batchIdx, channelIdx, heightIdx, widthIdx }));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return lossGradientForInput;
    }

    // def backprop(self, d_L_d_out):
    //     '''
    //     Performs a backward pass of the maxpool layer.
    //     Returns the loss gradient for this layer's inputs.
    //     - d_L_d_out is the loss gradient for this layer's outputs.
    //     '''
    //     d_L_d_input = np.zeros(self.last_input.shape)

    //     for im_region, i, j in self.iterate_regions(self.last_input):
    //       h, w, f = im_region.shape
    //       amax = np.amax(im_region, axis=(0, 1))

    //       for i2 in range(h):
    //         for j2 in range(w):
    //           for f2 in range(f):
    //             # If this pixel was the max value, copy the gradient to it.
    //             if im_region[i2, j2, f2] == amax[f2]:
    //               d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

    //     return d_L_d_input

    // pub fn call(self: *const Self, input: NDArray(f64, 3), nextAction: fn (NDArray(f64, 3), Allocator) !NDArray(f64, 3), allocator: Allocator) !NDArray(f64, 3) {
    //     var forwardRes = try self.forward(input, allocator);
    //     defer forwardRes.deinit();
    //     var output = try nextAction(forwardRes, allocator);
    //     return output;
    // }
};
