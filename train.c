#include "cmai.c"

int main(){ // main training loop. as of now train.c is mostly a test, it isnt what i want it to be yet. wait patiently until i finish my Break(TM)
    int vocab_size = 10, d_model = 16, num_blocks = 2, hidden_size = 64, seq_len = 4, stops = 1000; // training variables and parameters
    double learning_rate = 0.01; // also a training variable
    srand(47);

    embedding *embed = createEmbedding(vocab_size, d_model); // time to make the model structure
    transformer_network *net = createTransformerNetwork(num_blocks, d_model, hidden_size, relu); // this is too, part of the structure

    layer unembed;
    unembed.weights = newMatrix(d_model, vocab_size);
    unembed.biases = newMatrix(1, vocab_size);
    unembed.activation = linear; // doesn't matter, unembeddingForward doesnt use activation

    int i;
    for (i = 0; i < d_model * vocab_size; i++) { // randomize weights
        unembed.weights->data[i] = (rand() % 100) / 100.0 - 0.5;
    }

    for (int stop = 0; stop < stops; stop++) { // the main training cycle loop
        int input_tokens[] = {0, 1, 2, 3}; // data
        int target_tokens[] = {1, 2, 3, 4}; // what we want the AI to predict based on data

        matrix *embed_out = embeddingForward(embed, input_tokens, seq_len); // forward pass
        matrix *pe = positionalEncoding(seq_len, d_model);
        matrix *input_to_net = addMatrix(embed_out, pe);

        transformer_cache **caches = malloc(net->num_blocks * sizeof(transformer_cache *)); // allocate caches

        matrix *net_out = transformerNetworkForward(input_to_net, net, caches); // pass through the transformer blocks


        matrix *logits = unembeddingForward(net_out, &unembed); // pass through the unembedding layer to get probabilities

        double loss = crossEntropyLoss(logits, target_tokens); // That one ctrl alt delete comic

        if (stop % 100 == 0) {
            printf("stop %d - loss: %.6f\n", stop, loss); // prints Loss
        }

        matrix *output_grad = newMatrix(logits->rows, logits->columns); // backprop time. output gradient
        crossEntropyLossGradient(logits, target_tokens, output_grad); // calculate loss

        matrix *unembed_preact = newMatrix(net_out->rows, unembed.weights->columns); // I didn't cache this, but backward() needs it so
        matrix *net_out_grad = backward(&unembed, net_out, output_grad, learning_rate, unembed_preact); // this
        freeMatrix(unembed_preact); // frees matrix :face_holding_back_tears:

        matrix *current_grad = net_out_grad; // do the propogation
        for (int j = net->num_blocks - 1; j >= 0; j--) {
            matrix *next_grad = transformerBackward(net->blocks[j], caches[j], current_grad, learning_rate);
            freeMatrix(current_grad);
            current_grad = next_grad;
        }

        freeMatrix(output_grad); // Free the variables
        freeMatrix(current_grad); // the final gradient from the loop

        // Free the caches
        for (int j = 0; j < net->num_blocks; j++) {
            freeTransformerCache(caches[j]);
        }
        free(caches);

        // Free the forward pass matrices
        freeMatrix(embed_out); // fnuyy line number
        freeMatrix(pe);
        freeMatrix(input_to_net);
        freeMatrix(logits);
    }
}
