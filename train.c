#include "cmai.c"

void train(network *net, matrix *input, matrix *target, int stops, double learning_rate) { // train function. takes a network, an input matrix, a target matrix,
    int stop, i; // i chose stops instead of epochs because "epoch" sounds esoteric        // a stops integer (epochs if you're boring) and a learning rate
    for (stop = 0; stop < stops; stop++) { // the main training cycle loop

        // forward pass
        forward_cache *cache = networkForward(net, input); // makes a forward cache
        matrix *predicted = cache->layer_outputs[net->num_layers - 2]; // makes a predicted matrix

        // calculate loss
        double loss = mse(predicted, target); // https://cad-comic.com/comic/loss/

        // calculate output gradient (predicted - target)
        matrix *output_grad = newMatrix(predicted->rows, predicted->columns); // output gradient matrix

        for (i = 0; i < predicted->rows * predicted->columns; i++) // loopy loop loop for the gradient
            output_grad->data[i] = predicted->data[i] - target->data[i]; // yes

        networkBackward(net, cache, output_grad, learning_rate); // backward pass

        if (stop % 100 == 0) // loss printing loop
            printf("stop %d — loss: %.6f\n", stop, loss); // print loss every 100

        freeForwardCache(cache); // free beer
        freeMatrix(output_grad);
    }
}
