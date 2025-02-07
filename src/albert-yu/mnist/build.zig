const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardOptimizeOption(.{});
    const exe = b.addExecutable(.{
        .name = "mnist",
        .root_source_file = b.path("main.zig"),
        .target = target,
        .optimize = mode,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const run_linalg_tests = addUnitTestFile(b, "linalg.zig", target, mode);
    const run_layer_tests = addUnitTestFile(b, "layer.zig", target, mode);
    const run_maths_tests = addUnitTestFile(b, "maths.zig", target, mode);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_linalg_tests.step);
    test_step.dependOn(&run_layer_tests.step);
    test_step.dependOn(&run_maths_tests.step);
}

fn addUnitTestFile(b: *std.Build, filename: []const u8, target: std.Build.ResolvedTarget, mode: std.builtin.OptimizeMode) *std.Build.Step.Run {
    const exe_tests = b.addTest(.{
        .root_source_file = b.path(filename),
        .target = target,
        .optimize = mode,
    });

    const run_unit_tests = b.addRunArtifact(exe_tests);
    return run_unit_tests;
}
