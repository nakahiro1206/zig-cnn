const Conv2DTest = @import("layers/conv2dTest.zig").Conv2DTest;
test "Conv2D test" {
    try Conv2DTest();
}

const flattenTest = @import("layers/flatten.zig").flattenTest;
test "flatten test" {
    try flattenTest();
}

const maxPoolingTest = @import("layers/maxPoolingTest.zig").maxPoolingTest;
test "max pooling" {
    try maxPoolingTest();
}

const softmaxTest = @import("layers/softmaxTest.zig").softmaxTest;
test "softmax" {
    try softmaxTest();
}
