// https://github.com/albert-yu/mnist/blob/main/src/datapoint.zig
const std = @import("std");

fn copy_slice(src: []f64, dest: []f64) void {
    for (src, 0..) |elem, i| {
        dest[i] = elem;
    }
}

const randgen = std.rand.DefaultPrng;
var rand = randgen.init(0);

/// DataPoint struct of arrays
pub const DataPointSOA = struct {
    x: []f64,
    x_chunk_size: usize,
    y: []f64,
    y_chunk_size: usize,

    const Self = @This();

    pub fn len(self: Self) usize {
        return self.x.len / self.x_chunk_size;
    }

    pub fn x_at(self: Self, i: usize) []f64 {
        const start = i * self.x_chunk_size;
        return self.x[start .. start + self.x_chunk_size];
    }

    pub fn y_at(self: Self, i: usize) []f64 {
        const start = i * self.y_chunk_size;
        return self.y[start .. start + self.y_chunk_size];
    }

    pub fn slice_x(self: Self, i: usize, length: usize) []f64 {
        const start = i * self.x_chunk_size;
        return self.x[start .. start + length * self.x_chunk_size];
    }

    pub fn slice_y(self: Self, i: usize, length: usize) []f64 {
        const start = i * self.y_chunk_size;
        return self.y[start .. start + length * self.y_chunk_size];
    }

    pub fn shuffle(self: Self, allocator: std.mem.Allocator) !void {
        const size = self.len();
        var i: usize = 0;
        const x_temp = try allocator.alloc(f64, self.x_chunk_size);
        defer allocator.free(x_temp);
        const y_temp = try allocator.alloc(f64, self.y_chunk_size);
        defer allocator.free(y_temp);
        while (i < size) : (i += 1) {
            const random_offset = rand.random().int(usize);
            const new_i = (i + random_offset) % size;
            // swap
            const x_i = self.x_at(i);
            copy_slice(x_i, x_temp);
            const y_i = self.y_at(i);
            copy_slice(y_i, y_temp);

            copy_slice(self.x_at(new_i), x_i);
            copy_slice(self.y_at(new_i), y_i);

            copy_slice(x_temp, self.x_at(new_i));
            copy_slice(y_temp, self.y_at(new_i));
        }
    }
};
