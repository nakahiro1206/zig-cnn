const std = @import("std");

const VEC_SIZE = 2;

const Vec = @Vector(VEC_SIZE, f64);

const zero_vec: Vec = [_]f64{0} ** VEC_SIZE;

fn aligned_calloc(allocator: std.mem.Allocator, size: usize) ![]Vec {
    const ptr = try allocator.alignedAlloc(Vec, null, size);
    @memset(ptr, zero_vec);
    return ptr;
}

pub const Matrix = struct {
    data_vec: []Vec,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn new(rows: usize, cols: usize) Self {
        return Self{
            .rows = rows,
            .cols = cols,
            .data_vec = undefined,
        };
    }

    pub fn alloc(self: *Self, allocator: std.mem.Allocator) !void {
        const n_blocks = self.blocks_per_row();
        self.data_vec = try aligned_calloc(allocator, n_blocks * self.rows);
    }

    pub fn dealloc(self: Self, allocator: std.mem.Allocator) void {
        allocator.free(self.data_vec);
    }

    pub fn mul_alloc(self: *Self, allocator: std.mem.Allocator, right: Self) !Self {
        var result = Self.new(self.rows, right.cols);

        try result.alloc(allocator);
        try self.mul(allocator, right, &result);
        return result;
    }

    pub fn print(self: Self) void {
        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                const elem = self.at(i, j);
                std.debug.print("{} ", .{elem});
            }
            std.debug.print("\n", .{});
        }
        std.debug.print("\n", .{});
    }

    pub fn add(self: Self, other: Self, out: *Self) void {
        for (self.data_vec, 0..) |vec, i| {
            out.data_vec[i] = vec + other.data_vec[i];
        }
    }

    pub fn sub(self: Self, other: Self, out: *Self) void {
        for (self.data_vec, 0..) |vec, i| {
            out.data_vec[i] = vec - other.data_vec[i];
        }
    }

    /// Computes Hadamard product (element-wise multiplication)
    ///
    /// Assumes `out` is allocated to be the same length as both
    /// `vec1` and `vec2`.
    pub fn hadamard(self: Self, other: Self, out: *Self) void {
        for (self.data_vec, 0..) |vec, i| {
            out.data_vec[i] = vec * other.data_vec[i];
        }
    }

    /// Sets all elements to 0
    pub fn zeroes(self: Self) void {
        @memset(self.data_vec, zero_vec);
    }

    /// scales all matrix elements in-place
    pub fn scale(self: Self, scalar: f64) void {
        const scalar_vec: Vec = [_]f64{scalar} ** VEC_SIZE;
        for (self.data_vec, 0..) |elem, i| {
            self.data_vec[i] = elem * scalar_vec;
        }
    }

    /// Get number of blocks per row
    inline fn blocks_per_row(self: Self) usize {
        // This is just ceiling division
        return (self.cols + VEC_SIZE - 1) / VEC_SIZE;
    }

    /// Dumps the matrix data into a row-major slice
    pub fn dump(self: Self, allocator: std.mem.Allocator) ![]f64 {
        const data_raw = try allocator.alloc(f64, self.rows * self.cols);
        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                const elem = self.at(i, j);
                data_raw[i * self.cols + j] = elem;
            }
        }
        return data_raw;
    }

    /// Matrix multiplication, but assumes that the right matrix is already
    /// transposed.
    pub fn mul_t(self: *Self, right_t: Self, out: *Self) void {
        const n_blocks_l = self.blocks_per_row();
        const a = self.data_vec;
        const b = right_t.data_vec;
        const n_blocks_r = (right_t.cols + VEC_SIZE - 1) / VEC_SIZE;
        for (0..out.rows) |i| {
            for (0..out.cols) |j| {
                var acc = zero_vec;
                for (0..n_blocks_l) |k| {
                    const left_val = a[i * n_blocks_l + k];
                    const right_val = b[j * n_blocks_r + k];
                    acc += left_val * right_val;
                }

                var final: f64 = 0;
                for (0..VEC_SIZE) |k| {
                    final += acc[k];
                }
                out.set(i, j, final);
            }
        }
    }

    pub fn mul(self: *Self, allocator: std.mem.Allocator, right: Self, out: *Self) !void {
        // transpose right matrix for better cache locality
        const b = try right.t_alloc(allocator);
        defer b.dealloc(allocator);

        // perform the multiplication
        self.mul_t(b, out);
    }

    pub fn sub_alloc(self: Self, allocator: std.mem.Allocator, right: Self) !Self {
        var result = Self.new(self.rows, self.cols);
        try result.alloc(allocator);
        self.sub(right, &result);
        return result;
    }

    /// Transposes matrix and returns new one, which must be dealloc'd
    pub fn t_alloc(self: Self, allocator: std.mem.Allocator) !Self {
        var out = Self.new(self.cols, self.rows);
        try out.alloc(allocator);

        // swap rows and columns
        self.t(&out);
        return out;
    }

    pub fn t(self: Self, output: *Self) void {
        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                output.set(j, i, self.at(i, j));
            }
        }
    }

    pub fn for_each(self: Self, comptime op: fn (f64) f64) void {
        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                const op_result = op(self.at(i, j));
                self.set(i, j, op_result);
            }
        }
    }

    /// Returns the value at the given indices.
    ///
    /// Parameters:
    ///   i - 0-based row index
    ///   j - 0-based column index
    pub inline fn at(self: Self, i: usize, j: usize) f64 {
        const n_blocks = self.blocks_per_row();
        return self.data_vec[i * n_blocks + j / VEC_SIZE][j % VEC_SIZE];
    }

    /// Sets the value at the given indices.
    ///
    /// Parameters:
    ///   i - 0-based row index
    ///   j - 0-based column index
    ///   value - value to set
    pub fn set(self: Self, i: usize, j: usize, value: f64) void {
        const n_blocks = self.blocks_per_row();
        self.data_vec[i * n_blocks + j / VEC_SIZE][j % VEC_SIZE] = value;
    }

    /// Copies the input data into its own data buffer
    /// without checking bounds
    pub fn set_data(self: Self, data: []f64) void {
        for (0..(self.rows)) |i| {
            for (0..(self.cols)) |j| {
                const elem = data[i * self.cols + j];
                self.set(i, j, elem);
                // self.data_vec[i * n_blocks + j / VEC_SIZE][j % VEC_SIZE] = elem;
            }
        }
    }

    pub fn make_copy(self: Self, allocator: std.mem.Allocator) !Self {
        var copied = Self.new(self.rows, self.cols);
        try copied.alloc(allocator);
        for (self.data_vec, 0..) |vec, i| {
            copied.data_vec[i] = vec;
        }
        return copied;
    }

    pub fn assign(self: *Self, other: Self) void {
        self.rows = other.rows;
        self.cols = other.cols;
        for (other.data_vec, 0..) |vec, i| {
            self.data_vec[i] = vec;
        }
    }
};

const err_tolerance = 1e-9;

test "transpose test" {
    const allocator = std.testing.allocator;
    var matrix_data = [_]f64{
        1, 2, 3,
        4, 5, 6,
    };
    var matrix = Matrix.new(2, 3);
    try matrix.alloc(allocator);
    matrix.set_data(&matrix_data);
    defer matrix.dealloc(allocator);
    const t_matrix = try matrix.t_alloc(allocator);
    defer t_matrix.dealloc(allocator);
    const expected_rows: usize = 3;
    const expected_cols: usize = 2;
    try std.testing.expectEqual(expected_rows, t_matrix.rows);
    try std.testing.expectEqual(expected_cols, t_matrix.cols);
    var result_data = [_]f64{
        1, 4,
        2, 5,
        3, 6,
    };
    const data_raw = try t_matrix.dump(allocator);
    defer allocator.free(data_raw);
    try std.testing.expectEqualSlices(f64, &result_data, data_raw);
}

test "matrix multiplication test" {
    const allocator = std.testing.allocator;
    const mat_t = f64;
    var data = [_]mat_t{
        1, 2, 3,
        3, 1, 4,
    };
    var data_other = [_]mat_t{
        1, 1,
        2, 1,
        2, 5,
    };
    var matrix = Matrix.new(2, 3);
    try matrix.alloc(allocator);
    matrix.set_data(&data);
    defer matrix.dealloc(allocator);

    var matrix_other = Matrix.new(3, 2);
    try matrix_other.alloc(allocator);
    matrix_other.set_data(&data_other);
    defer matrix_other.dealloc(allocator);
    var out_matrix = Matrix.new(2, 2);
    try out_matrix.alloc(allocator);
    defer out_matrix.dealloc(allocator);

    try matrix.mul(allocator, matrix_other, &out_matrix);
    var expected_out_data = [_]mat_t{
        11, 18,
        13, 24,
    };
    const out_data = try out_matrix.dump(allocator);
    defer allocator.free(out_data);
    try std.testing.expectEqualSlices(mat_t, &expected_out_data, out_data);
}

test "outer product test" {
    const mat_t = f64;
    const allocator = std.testing.allocator;
    var data = [_]mat_t{
        1,
        2,
        3,
        4,
    };
    var data_other = [_]mat_t{
        1, 2, 3,
    };
    var matrix = Matrix.new(4, 1);
    try matrix.alloc(allocator);
    matrix.set_data(&data);
    defer matrix.dealloc(allocator);

    var matrix_other = Matrix.new(1, 3);
    try matrix_other.alloc(allocator);
    matrix_other.set_data(&data_other);
    defer matrix_other.dealloc(allocator);

    var out_matrix = try matrix.mul_alloc(allocator, matrix_other);
    defer out_matrix.dealloc(allocator);
    var expected_out_data = [_]mat_t{
        1, 2, 3,
        2, 4, 6,
        3, 6, 9,
        4, 8, 12,
    };
    const out_data = try out_matrix.dump(allocator);
    defer allocator.free(out_data);
    try std.testing.expectEqualSlices(mat_t, &expected_out_data, out_data);
}

test "inner product test" {
    const allocator = std.testing.allocator;
    var data_a_t = [_]f64{
        1, 2, 3,
    };
    var a = Matrix.new(3, 1);
    try a.alloc(allocator);
    a.set_data(&data_a_t);
    defer a.dealloc(allocator);

    var a_t = Matrix.new(1, 3);
    try a_t.alloc(allocator);
    a_t.set_data(&data_a_t);
    defer a_t.dealloc(allocator);

    var out = try a_t.mul_alloc(allocator, a);
    defer out.dealloc(allocator);
    var expected_out_data = [_]f64{14};
    const out_data = try out.dump(allocator);
    defer allocator.free(out_data);
    try std.testing.expectEqualSlices(f64, &expected_out_data, out_data);
}

test "linear transform test with allocation" {
    const allocator = std.testing.allocator;
    var identity_matrix_data = [_]f64{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    };
    var identity_matrix = Matrix.new(3, 3);
    try identity_matrix.alloc(allocator);
    defer identity_matrix.dealloc(allocator);
    identity_matrix.set_data(&identity_matrix_data);

    var vector_data = [_]f64{
        4,
        0,
        223,
    };
    var some_vector = Matrix.new(3, 1);
    try some_vector.alloc(allocator);
    defer some_vector.dealloc(allocator);
    some_vector.set_data(&vector_data);

    const result = try identity_matrix.mul_alloc(allocator, some_vector);
    defer result.dealloc(allocator);

    const result_data = try result.dump(allocator);
    defer allocator.free(result_data);

    try std.testing.expectEqualSlices(f64, result_data, &vector_data);
}
