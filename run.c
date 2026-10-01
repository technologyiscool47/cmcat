#include "misc.c" // misc includes literally everything

int main() {
    int merges_a[2000], merges_b[2000]; int num_merges;
    loadBPE("tokenizer.bin", merges_a, merges_b, &num_merges); // Load BPE before creating the model variables. trust

    int vocab_size = 256 + num_merges, d_model = 64, num_blocks = 2, hidden_size = 64, seq_len = 16, stops = 20000; // creating model variables
    srand(47); // set random seed

    embedding *embed = createEmbedding(vocab_size, d_model); // model architecture stuffembedding stuff
    transformer_network *net = createTransformerNetwork(num_blocks, d_model, hidden_size, relu); // creates the network

    layer unembed; // unembed things
    unembed.weights = newMatrix(d_model, vocab_size);
    unembed.biases = newMatrix(1, vocab_size);
    unembed.activation = linear;

    loadTransformerModel("model.bin", embed, net, &unembed); // loads the model

    const char *prompt = "GNU GENERAL PUBLIC LICENSE"; // generation
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

    for (int step = 0; step < 50; step++) {
        matrix *embed_out = embeddingForward(embed, input_tokens, seq_len); // forward pass (which i HATE)
        matrix *pe = positionalEncoding(seq_len, d_model);
        matrix *input_to_net = addMatrix(embed_out, pe);

        transformer_cache **caches = malloc(net->num_blocks * sizeof(transformer_cache *));
        matrix *net_out = transformerNetworkForward(input_to_net, net, caches);
        matrix *logits = unembeddingForward(net_out, &unembed);

        // --- SAMPLING (Instead of Argmax) ---
        double r = (double)rand() / RAND_MAX; // Roll a dice between 0 and 1
        double cumulative_prob = 0.0;
        int best_token = 0; // Fallback

        // Walk through the probabilities until we hit our random number
        for (int j = 0; j < vocab_size; j++) {
            cumulative_prob += getVal(logits, seq_len - 1, j);
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

        for (int i = 0; i < seq_len - 1; i++) { // shifts array to the left
            input_tokens[i] = input_tokens[i + 1];
        }
        input_tokens[seq_len - 1] = best_token;

        freeMatrix(embed_out); // DOWNLOAD FREE RAM
        freeMatrix(pe);
        freeMatrix(input_to_net);
        for (int i = 0; i < net->num_blocks; i++) {
            freeTransformerCache(caches[i]);
        }
        free(caches);
        freeMatrix(logits);
    }
}
