const std = @import("std");
const expect = std.testing.expect;

fn relu(x: f32) f32 {
    return if (x > 0) x else 0;
}

pub fn softmax(input: []f32) []f32 {
    var exp_sum: f32 = 0.0;
    for (input) |x| exp_sum += @exp(x);

    // &input: reference to [] f32. By default, the reference is immutable.
    // input: mutable [] f32
    for (input) |*x| {
        x.* = @exp(x.*) / exp_sum;
    }
    return input;
}

test "relu" {
    const x = 1.0;
    const y = relu(x);
    try expect(y == 1.0);
}

test "softmax" {
    var input = [_]f32{ 1.0, 2.0, 3.0 };
    const output = softmax(&input);
    try expect(@abs(output[0] - 0.09003057) < 0.0001);
    try expect(@abs(output[1] - 0.24472848) < 0.0001);
    try expect(@abs(output[2] - 0.66524094) < 0.0001);
}
