const std = @import("std");
const linalg = @import("linalg.zig");
const maths = @import("maths.zig");

pub fn Gradients(comptime IN: usize, comptime OUT: usize) type {
    return struct {
        weights: linalg.Matrix,
        biases: linalg.Matrix,

        const Self = @This();

        pub fn alloc(allocator: std.mem.Allocator) !Self {
            var w = linalg.Matrix.new(OUT, IN);
            try w.alloc(allocator);
            var b = linalg.Matrix.new(OUT, 1);
            try b.alloc(allocator);
            return Self{ .weights = w, .biases = b };
        }

        pub fn dealloc(self: Self, allocator: std.mem.Allocator) void {
            self.biases.dealloc(allocator);
            self.weights.dealloc(allocator);
        }
    };
}

/// Single layer in neural net
pub fn Layer(comptime IN: usize, comptime OUT: usize) type {
    return struct {
        weights: linalg.Matrix,
        biases: linalg.Matrix,
        last_input: linalg.Matrix,
        last_z: linalg.Matrix,

        forward_cache: linalg.Matrix,
        gradient_results: Gradients(IN, OUT),
        z_changes: linalg.Matrix,
        last_input_t: linalg.Matrix,

        const Self = @This();

        pub fn alloc(allocator: std.mem.Allocator) !Self {
            var w = linalg.Matrix.new(OUT, IN);
            try w.alloc(allocator);
            var b = linalg.Matrix.new(OUT, 1);
            try b.alloc(allocator);
            var last_z = linalg.Matrix.new(OUT, 1);
            try last_z.alloc(allocator);
            var forward_cache = linalg.Matrix.new(OUT, 1);
            try forward_cache.alloc(allocator);

            var z_changes = linalg.Matrix.new(OUT, 1);
            try z_changes.alloc(allocator);
            const gradient_results = try Gradients(IN, OUT).alloc(allocator);
            var last_input_t = linalg.Matrix.new(1, IN);
            try last_input_t.alloc(allocator);

            return Self{
                .last_input = undefined,
                .forward_cache = forward_cache,
                .weights = w,
                .biases = b,
                .last_z = last_z,
                .gradient_results = gradient_results,
                .z_changes = z_changes,
                .last_input_t = last_input_t,
            };
        }

        pub fn dealloc(self: Self, allocator: std.mem.Allocator) void {
            self.weights.dealloc(allocator);
            self.biases.dealloc(allocator);
            self.last_z.dealloc(allocator);
            self.forward_cache.dealloc(allocator);
            self.gradient_results.dealloc(allocator);
            self.z_changes.dealloc(allocator);
            self.last_input_t.dealloc(allocator);
        }

        /// Caller does not need to free result--it is owned by the layer
        pub fn forward(self: *Self, allocator: std.mem.Allocator, input: linalg.Matrix, comptime activation_fn: fn (f64) f64) !linalg.Matrix {
            try self.weights.mul(allocator, input, &self.forward_cache);
            self.forward_cache.add(self.biases, &self.forward_cache);
            self.last_z.assign(self.forward_cache);
            self.forward_cache.for_each(activation_fn);
            self.last_input = input;
            return self.forward_cache;
        }

        /// Caller does not need to free result--it is owned by the layer
        pub fn backward(self: Self, allocator: std.mem.Allocator, err: linalg.Matrix, comptime activation_prime: fn (f64) f64) !Gradients(IN, OUT) {
            var gradient_results = self.gradient_results;
            var z_changes = self.z_changes;
            z_changes.assign(self.last_z);
            z_changes.for_each(activation_prime);
            z_changes.hadamard(err, &gradient_results.biases);

            var last_input_t = self.last_input_t;
            self.last_input.t(&last_input_t);
            try gradient_results.biases.mul(allocator, last_input_t, &gradient_results.weights);
            return gradient_results;
        }

        pub fn init_randn(self: Self) void {
            self.weights.for_each(get_randn);
            self.biases.for_each(get_randn);
        }

        pub fn init_zeros(self: Self) void {
            self.weights.zeroes();
            self.biases.zeroes();
        }
    };
}

const randgen = std.rand.DefaultPrng;
var rand = randgen.init(1);

fn get_randn(_: f64) f64 {
    return rand.random().floatNorm(f64);
}

test "feedforward test" {
    const allocator = std.testing.allocator;
    var layer1 = try Layer(2, 2).alloc(allocator);
    defer layer1.dealloc(allocator);
    var layer2 = try Layer(2, 2).alloc(allocator);
    defer layer2.dealloc(allocator);

    var w_1 = [_]f64{
        1, 0,
        0, 1,
    };
    var b_1 = [_]f64{
        0.5,
        0.5,
    };
    layer1.weights.set_data(&w_1);
    layer1.biases.set_data(&b_1);

    var w_2 = [_]f64{
        -1, 0,
        0,  1,
    };
    var b_2 = [_]f64{
        0.2,
        0.2,
    };

    layer2.weights.set_data(&w_2);
    layer2.biases.set_data(&b_2);
    var input_x = [_]f64{
        0.1,
        0.1,
    };
    var input = linalg.Matrix.new(2, 1);
    try input.alloc(allocator);
    defer input.dealloc(allocator);
    input.set_data(&input_x);
    const TOLERANCE = 1e-9;

    const result1 = try layer1.forward(allocator, input, maths.sigmoid);
    const result2 = try layer2.forward(allocator, result1, maths.sigmoid);

    // var output = network.output_layer();
    const expected_out = [_]f64{ 0.3903940131009935, 0.6996551604890665 };
    const activation = result2;
    const activation_res = try activation.dump(allocator);
    defer allocator.free(activation_res);
    try std.testing.expectApproxEqRel(expected_out[0], activation_res[0], TOLERANCE);
    try std.testing.expectApproxEqRel(expected_out[1], activation_res[1], TOLERANCE);
}

test "backpropagation test" {
    const allocator = std.testing.allocator;
    var x_data = [_]f64{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126, 136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253, 225, 172, 253, 242, 195, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 253, 253, 253, 251, 93, 82, 82, 56, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198, 182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0, 43, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171, 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 172, 226, 253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    var x = linalg.Matrix.new(x_data.len, 1);
    try x.alloc(allocator);
    defer x.dealloc(allocator);
    x.set_data(&x_data);

    var y_data = [_]f64{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
    var y = linalg.Matrix.new(y_data.len, 1);
    try y.alloc(allocator);
    defer y.dealloc(allocator);
    y.set_data(&y_data);
    const image_size = 28 * 28;
    const HIDDEN_LAYER_SIZE = 30;
    const DIGITS = 10;
    var hidden_layer = try Layer(image_size, HIDDEN_LAYER_SIZE).alloc(allocator);
    defer hidden_layer.dealloc(allocator);
    hidden_layer.init_zeros();

    var output_layer = try Layer(HIDDEN_LAYER_SIZE, DIGITS).alloc(allocator);
    defer output_layer.dealloc(allocator);
    output_layer.init_zeros();

    // feedforward
    const result1 = try hidden_layer.forward(allocator, x, maths.sigmoid);
    const result2 = try output_layer.forward(allocator, result1, maths.sigmoid);

    // compute error
    var err = try result2.sub_alloc(allocator, y);
    defer err.dealloc(allocator);

    // backward
    const grad1 = try output_layer.backward(allocator, err, maths.sigmoid_prime);

    var expected_delta_b = [_]f64{ 0.125, 0.125, 0.125, 0.125, 0.125, -0.125, 0.125, 0.125, 0.125, 0.125 };
    const actual_delta_b = try grad1.biases.dump(allocator);
    defer allocator.free(actual_delta_b);
    try std.testing.expectEqualSlices(f64, &expected_delta_b, actual_delta_b);

    var expected_delta_w = [_]f64{
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  -0.0625, -0.0625, -0.0625,
        -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625,
        -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625,
        -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
        0.0625,  0.0625,  0.0625,
    };
    const weights_result = try grad1.weights.dump(allocator);
    defer allocator.free(weights_result);
    try std.testing.expectEqualSlices(f64, &expected_delta_w, weights_result);
}
