/*
 * cmcat - local llm, from scratch, in C.
 * Copyright (C) 2026 technologyiscool47
 *
 * This program is free software (both as in free beer and as in freedom):
 * you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License v3.0 as published by the Free Software Foundation.
 *
 * Commercial licensing is available. technologyiscool47 is active on the following platforms:
 * - Neocities (https://technologyiscool47.neocities.com)
 * - GitHub (https://github.com/technologyiscool47)
 * - E-mail (technologyiscool47@gmail.com)
 * Contact the author for details.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License v3.0 for more details.
 */

#include "misc.c" // misc includes literally everything

int main() {
    int merges_a[2000], merges_b[2000]; int num_merges;
    loadBPE("tokenizer.bin", merges_a, merges_b, &num_merges); // Load BPE before creating the model variables. trust

    int vocab_size = 256 + num_merges, d_model = 128, num_blocks = 3, hidden_size = 256, seq_len = 16, stops = 20000; // creating model variables
    srand(47); // set random seed

    embedding *embed = createEmbedding(vocab_size, d_model); // model architecture stuffembedding stuff
    transformer_network *net = createTransformerNetwork(num_blocks, d_model, hidden_size, relu); // creates the network

    layer unembed; // unembed things
    unembed.weights = newPersistentMatrix(d_model, vocab_size);
    unembed.biases = newPersistentMatrix(1, vocab_size);
    unembed.activation = linear;

    loadTransformerModel("model.bin", embed, net, &unembed); // loads the model

    unembed.packed_weights = NULL;
    unembed.unpacked_weights = NULL;
    for (int i = 0; i < net->num_blocks; i++) { // repack transformer blocks
        quantizeAndPack(net->blocks[i]->wq.weights, &net->blocks[i]->wq.packed_weights, &net->blocks[i]->wq.scale);
        quantizeAndPack(net->blocks[i]->wk.weights, &net->blocks[i]->wk.packed_weights, &net->blocks[i]->wk.scale);
        quantizeAndPack(net->blocks[i]->wv.weights, &net->blocks[i]->wv.packed_weights, &net->blocks[i]->wv.scale);
        quantizeAndPack(net->blocks[i]->wo.weights, &net->blocks[i]->wo.packed_weights, &net->blocks[i]->wo.scale);
        quantizeAndPack(net->blocks[i]->ff_w_gate.weights, &net->blocks[i]->ff_w_gate.packed_weights, &net->blocks[i]->ff_w_gate.scale);
        quantizeAndPack(net->blocks[i]->ff_w_up.weights, &net->blocks[i]->ff_w_up.packed_weights, &net->blocks[i]->ff_w_up.scale);
        quantizeAndPack(net->blocks[i]->ff_w_down.weights, &net->blocks[i]->ff_w_down.packed_weights, &net->blocks[i]->ff_w_down.scale);
    }

    // allocates KV caches for each block
    int head_dim = d_model / NUM_HEADS;
    int kv_dim = head_dim * NUM_KV_GROUPS;

    kv_cache **kv_caches = malloc(net->num_blocks * sizeof(kv_cache *)); // allocates
    for (int i = 0; i < net->num_blocks; i++) { // i
        kv_caches[i] = malloc(sizeof(kv_cache)); // allocates
        kv_caches[i]->K = newPersistentMatrix(MAX_CONTEXT, kv_dim); // matrices
        kv_caches[i]->V = newPersistentMatrix(MAX_CONTEXT, kv_dim);
        kv_caches[i]->len = 0; // len
    }

    const char *prompt = "Once upon a time, "; // generation
    long prompt_len;
    int *tokens = tokenize(prompt, &prompt_len, merges_a, merges_b, num_merges);

    int input_tokens[seq_len]; // initial 4 tokens
    for (int i = 0; i < seq_len; i++) {
        int idx = prompt_len - seq_len + i;
        if (idx >= 0) {
            input_tokens[i] = tokens[idx];
        } else {
            input_tokens[i] = 32; // Padding
        }
    }
    free(tokens); // Don't need the full array anymore

    printf("Generating text...\n");

    initArena();
    for (int step = 0; step < 50; step++) {
        resetArena();

        matrix *logits;
        matrix *net_out;
        int current_token;

        if (step == 0) {
            // Prompt pass
            matrix *embed_out = embeddingForward(embed, input_tokens, seq_len);
            transformer_cache **caches = malloc(net->num_blocks * sizeof(transformer_cache *));
            net_out = transformerNetworkForward(embed_out, net, caches);
            logits = unembeddingForward(net_out, &unembed);

            // Copy K and V into the persistent KV Cache
            for (int b = 0; b < net->num_blocks; b++) {
                for (int r = 0; r < seq_len; r++) {
                    for (int c = 0; c < kv_dim; c++) {
                        kv_caches[b]->K->data[r * kv_dim + c] = caches[b]->K->data[r * kv_dim + c];
                        kv_caches[b]->V->data[r * kv_dim + c] = caches[b]->V->data[r * kv_dim + c];
                    }
                }
                kv_caches[b]->len = seq_len;
            }

            // Cleanup prompt pass
            freeMatrix(embed_out);
            for (int b = 0; b < net->num_blocks; b++) freeTransformerCache(caches[b]);
            free(caches);
        } else {
            // Generation pass
            current_token = input_tokens[seq_len - 1]; // The token generated in the previous step

            int single_token[1] = {current_token};
            matrix *embed_out = embeddingForward(embed, single_token, 1);

            net_out = transformerNetworkForwardStep(embed_out, net, kv_caches);
            logits = unembeddingForward(net_out, &unembed);

        }

        // Mask out untrained high bytes (128-255)
        for (int j = 128; j < 256; j++) setVal(logits, 0, j, 0.0f);
        float prob_sum = 0.0f;
        for (int j = 0; j < vocab_size; j++) prob_sum += getVal(logits, 0, j);
        for (int j = 0; j < vocab_size; j++) setVal(logits, 0, j, getVal(logits, 0, j) / prob_sum);

        // Sampling
        double r = (double)rand() / RAND_MAX;
        double cumulative_prob = 0.0;
        int best_token = 0;

        for (int j = 0; j < vocab_size; j++) {
            cumulative_prob += getVal(logits, 0, j);
            if (r <= cumulative_prob) {
                best_token = j;
                break;
            }
        }

        printf("Token: %d\n", best_token);

        unsigned char text_buf[1000]; // decode the token to text
        int buf_idx = 0;
        expandToken(best_token, merges_a, merges_b, num_merges, text_buf, &buf_idx);
        text_buf[buf_idx] = '\0'; // Null terminate
        printf("%s\n", text_buf);

        // Shift the array
        for (int i = 0; i < seq_len - 1; i++) {
            input_tokens[i] = input_tokens[i + 1];
        }
        input_tokens[seq_len - 1] = best_token;

        freeMatrix(logits);
        freeMatrix(net_out);
    }
}
