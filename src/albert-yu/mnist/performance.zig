const std = @import("std");

pub const Stopwatch = struct {
    last_ts: i128,
    enabled: bool,

    pub fn new() Stopwatch {
        return Stopwatch{
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

    pub fn setEnabled(self: *Stopwatch, enabled: bool) void {
        self.enabled = enabled;
    }
};
