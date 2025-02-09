## Zig CNN

Naive CNN model implemented in Zig.

### Performance for [KMIST](https://github.com/rois-codh/kmnist)

I referred to the code of [CNN from scratch](https://github.com/vzhou842/cnn-from-scratch) for constructing the model, and I compared the performance and accuracy with his model and another model he implemented as a benchmark.

Uniformed condition: epochs = 1, batch size = 1, learning rate = 0.005.

|Model| Training Loss | Training Accuracy | Evaluation Loss | Evaluation Accuracy | Training Time (sec) |
| ----- | --------------| ------------------|-----------------|---------------------|---------------|
|CNN from scratch(numpy only) | 0.433  | 0.87 | 0.785 | 0.768 | 1139 |
|CNN from scratch(keras)      | 0.6455 | 0.8018 | 0.6837 | 0.7977 | 19 |
|My model                     | ..     | 0.7329 | 1.659 | 0.674 | 196 |

### Referred Repository
- [pblischak/zig-ndarray](https://github.com/pblischak/zig-ndarray)
- [albert-yu/mnist](https://github.com/albert-yu/mnist)
- [vzhou842/cnn-from-scratch](https://github.com/vzhou842/cnn-from-scratch)
- [rois-codh/kmnist](https://github.com/rois-codh/kmnist)
