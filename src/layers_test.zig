const Conv2DTest = @import("layers/test/conv2dTest.zig").Conv2DTest;
test "Conv2D test" {
    try Conv2DTest();
}

const flattenTest = @import("layers/test/flattenTest.zig").flattenTest;
test "flatten test" {
    try flattenTest();
}

const maxPoolingTest = @import("layers/test/maxPoolingTest.zig").maxPoolingTest;
test "max pooling" {
    try maxPoolingTest();
}

const softmaxTest = @import("layers/test/softmaxTest.zig").softmaxTest;
test "softmax" {
    try softmaxTest();
}
