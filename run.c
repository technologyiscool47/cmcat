#include "cmai.c"

int main() {
    // 1. create network with same structure as training
    int sizes[] = {2, 3, 1};
    int num_layers = 3;
    network *net = createNetwork(sizes, num_layers, sigmoid);

    // 2. load weights
    loadNetwork(net, "test123456.bin");

    // 3. create input
    matrix *input = newMatrix(1, 2);
    setVal(input, 0, 0, 0.5);
    setVal(input, 0, 1, 0.5);

    // 4. forward pass
    forward_cache *cache = networkForward(net, input);
    matrix *output = cache->layer_outputs[net->num_layers - 2];

    // 5. print output
    printf("Output: %.6f\n", output->data[0]);

    // 6. free
    freeForwardCache(cache);
    freeMatrix(input);
    freeNetwork(net);

    return 0;
}
