const std = @import("std");

pub const Stopwatch = struct {
    last_ts: i128,
    init_ts: i128,
    enabled: bool,

    pub fn new() Stopwatch {
        return Stopwatch{
            .init_ts = std.time.nanoTimestamp(),
            .last_ts = 0,
            .enabled = true,
        };
    }

    pub fn start(self: *Stopwatch) void {
        if (!self.enabled) {
            return;
        }
        self.last_ts = std.time.nanoTimestamp();
    }

    pub fn report(self: *Stopwatch, label: []const u8) void {
        if (!self.enabled) {
            return;
        }
        const elapsed = std.time.nanoTimestamp() - self.last_ts;
        std.debug.print("elapsed: {}ns ({s})\n", .{ elapsed, label });
        self.last_ts = std.time.nanoTimestamp();
    }

    pub fn reportForHeadding(self: *Stopwatch) !void {
        if (!self.enabled) {
            return;
        }
        const nsAsi128: i128 = std.time.nanoTimestamp() - self.init_ts;
        const ns = @as(f64, @floatFromInt(nsAsi128));
        const sec: f64 = ns / 1e9;
        var buf: [6]u8 = undefined;
        const str = try std.fmt.bufPrint(&buf, "{d:.2}", .{sec});
        std.debug.print("{s: >6} sec | ", .{str});
    }

    pub fn setEnabled(self: *Stopwatch, enabled: bool) void {
        self.enabled = enabled;
    }
};
