/*
 * cmcat - local llm, from scratch, in C.
 * Copyright (C) 2026 technologfyiscool47
 *
 * This program is free software (both as in free beer and as in freedom):
 * you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License v3.0 as published by the Free Software Foundation.
 *
 * Commercial licensinfg is available. technologfyiscool47 is active on the following platforms:
 * - Neocities (https://technologfyiscool47.neocities.com)
 * - GitHub (https://github.com/technologfyiscool47)
 * - E-mail (technologfyiscool47@gmail.com)
 * Contact the author for details.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License v3.0 for more details.
 */

#include "misc.c" // includes literally everything train.c needs

int main(){ // main training loop. as of now train.c is mostly a test, it isnt what i want it to be yet. wait patiently
    // omp_set_num_threads(omp_get_num_procs());
    int merges_a[2000], merges_b[2000]; int num_merges; // tokenization. this sets up the merges
    trainBPE("LICENSE", 500, merges_a, merges_b, &num_merges); // trains and saves bpe on the license
    saveBPE("tokenizer.bin", merges_a, merges_b, num_merges); // saves bpe

    int vocab_size = 256 + num_merges, d_model = 64, num_blocks = 2, hidden_size = 64, seq_len = 16, stops = 20000; // training variables and parameters
    float learning_rate = 0.001; // also a training variable
    srand(47);

    embedding *embed = createEmbedding(vocab_size, d_model); // time to make the model structure. no more tokenization for now
    transformer_network *net = createTransformerNetwork(num_blocks, d_model, hidden_size, relu); // this is too, part of the structure

    layer unembed;
    unembed.weights = newPersistentMatrix(d_model, vocab_size);
    unembed.biases = newPersistentMatrix(1, vocab_size);
    unembed.activation = linear; // doesn't matter, unembeddingForward doesnt use activation
    unembed.weight_adam = createAdamState(unembed.weights);
    unembed.bias_adam = createAdamState(unembed.biases  );

    float scale = 1.0f / sqrtf(d_model);
    int i;
    for (i = 0; i < d_model * vocab_size; i++) { // randomize weights
        unembed.weights->data[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale;
    }

    unembed.packed_weights = NULL;
    unembed.unpacked_weights = NULL;
    quantizeAndPack(unembed.weights, &unembed.packed_weights, &unembed.scale);
    unembed.unpacked_weights = unpackWeights(unembed.packed_weights, unembed.scale, unembed.weights->rows, unembed.weights->columns);

    long file_size; // tokenization. time to set up bpe
    unsigned char *text_bytes = readFileToBytes("LICENSE", &file_size); // Read the file to bytes

    long total_tokens;
    int *text_tokens = tokenize((const char*)text_bytes, &total_tokens, merges_a, merges_b, num_merges); // Tokenize the text!
    free(text_bytes); // free the bytes because we have tokens

    printf("Total tokens in training data: %ld\n", total_tokens);

    int input_tokens[seq_len];
    int target_tokens[seq_len];

    initArena();
    for (int stop = 0; stop < stops; stop++) { // the main training cycle loop
        resetArena();
        // Pick a random starting point in the license
        int start = rand() % (total_tokens - seq_len - 1);

        // Grab our input sequence and our target sequence (shifted by 1)
        for (int k = 0; k < seq_len; k++) {
            input_tokens[k] = text_tokens[start + k];
            target_tokens[k] = text_tokens[start + k + 1];
        }

        matrix *embed_out = embeddingForward(embed, input_tokens, seq_len); // forward pass

        transformer_cache **caches = malloc(net->num_blocks * sizeof(transformer_cache *)); // allocate caches
        matrix *net_out = transformerNetworkForward(embed_out, net, caches); // pass through the transformer block
        matrix *logits = unembeddingForward(net_out, &unembed); // pass through the unembedding layer to get probabilities

        float loss = crossEntropyLoss(logits, target_tokens); // That one ctrl alt delete comic

        if (stop % 100 == 0) {
            printf("stop %d - loss: %.6f\n", stop, loss); // prints Loss
        }

        matrix *output_grad = newMatrix(logits->rows, logits->columns); // backprop time. output gradient
        crossEntropyLossGradient(logits, target_tokens, output_grad); // calculate loss
        clipGradients(output_grad, 1.0f); // Don't let the gradients explode!

        matrix *unembed_preact = newMatrix(net_out->rows, unembed.weights->columns); // I didn't cache this, but backward() needs it so
        matrix *net_out_grad = backward(&unembed, net_out, output_grad, learning_rate, unembed_preact); // this
        freeMatrix(unembed_preact); // frees matrix :face_holding_back_tears:

        matrix *current_grad = net_out_grad; // do the propogation // fnuyy line nuber
        for (int j = net->num_blocks - 1; j >= 0; j--) {
            matrix *next_grad = transformerBackward(net->blocks[j], caches[j], current_grad, learning_rate);
            clipGradients(next_grad, 1.0f); // Clip internal gradients because i didnt think of implementing global clipping and now i have to either do local or rewrite everything smh
            freeMatrix(current_grad);
            current_grad = next_grad;
        }

        embeddingBackward(embed, input_tokens, seq_len, current_grad, learning_rate); // Backward on embeddings. i forgot to do this

        freeMatrix(output_grad); // Free the variables
        freeMatrix(current_grad); // the final gradient from the loop

        // Free the caches
        for (int j = 0; j < net->num_blocks; j++) {
            freeTransformerCache(caches[j]);
        }
        free(caches);

        // Free the forward pass matrices
        freeMatrix(embed_out);
        freeMatrix(logits);
    }
    saveTransformerModel("model.bin", embed, net, &unembed);
    printf("Model saved to model.bin!\n");
}
