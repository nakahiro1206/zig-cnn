const std = @import("std");
const mnist = @import("externalResources/mnist.zig");
const read_file = @import("externalResources/utils.zig").read_file;

const NDArray = @import("pblischak/zig-ndarray/ndarray.zig").NDArray;

pub fn main() !void {
    // Prints to stderr (it's a shortcut based on `std.io.getStdErr()`)
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});

    // stdout is for the actual output of your application, for example if you
    // are implementing gzip, then only the compressed bytes should be sent to
    // stdout, not any debugging messages.
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    try stdout.print("Run `zig build test` to run the tests.\n", .{});

    // load MNIST dataset.
    const TRAIN_LABELS_FILE = "data/train-labels-idx1-ubyte";
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        if (status == .leak) {
            std.debug.panic("got leak", .{});
        }
    }
    const allocator = gpa.allocator();
    const train_labels_buffer = try read_file(allocator, TRAIN_LABELS_FILE);
    defer allocator.free(train_labels_buffer);

    // read training labels
    const start_index = 8;
    const labels = train_labels_buffer[start_index..];

    // read training images
    const TRAIN_IMAGES_FILE = "data/train-images-idx3-ubyte";
    const train_images_buffer = try read_file(allocator, TRAIN_IMAGES_FILE);
    defer allocator.free(train_images_buffer);

    // can read image count from file, but should be exactly the same as labels
    const img_start_offset = 16;
    const images = train_images_buffer[img_start_offset..];

    // read test images
    const TEST_IMAGES_FILE = "data/t10k-images-idx3-ubyte";
    const test_images_buffer = try read_file(allocator, TEST_IMAGES_FILE);
    defer allocator.free(test_images_buffer);
    const test_images = test_images_buffer[img_start_offset..];

    // read test labels
    const TEST_LABELS_FILE = "data/t10k-labels-idx1-ubyte";
    const test_labels_buffer = try read_file(allocator, TEST_LABELS_FILE);
    defer allocator.free(test_labels_buffer);
    const test_labels = test_labels_buffer[start_index..];

    const image_size: comptime_int = 28 * 28;

    const DIGITS = 10;
    std.debug.print("making training data points...", .{});
    const train_data_points = try mnist.make_mnist_data_points_soa(allocator, images, image_size, labels, DIGITS);
    defer mnist.free_mnist_data_points_soa(allocator, train_data_points);
    std.debug.print("made {} train data points.\n", .{train_data_points.len()});

    std.debug.print("making test data points...", .{});
    const test_data = try mnist.make_mnist_data_points_soa(allocator, test_images, image_size, test_labels, DIGITS);
    defer mnist.free_mnist_data_points_soa(allocator, test_data);
    std.debug.print("made {} test data points.\n", .{test_data.len()});

    // const HIDDEN_LAYER_SIZE = 30;
    // const ETA = 0.05;
    // const EPOCHS = 100;

    try bw.flush(); // don't forget to flush!
}
