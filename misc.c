#include <stdio.h>
#include <stdlib.h>
#include <string.h> // memset
#include "cmai.c"

#define MAX_VOCAB 2000 // max vocab token size

/* ------------------ *\
   | Byte Level BPE |
\* ------------------ */

unsigned char *readFileToBytes(const char *filename, long *file_size){ // reads a file to bytes
    FILE *file = fopen(filename, "rb"); // opens file
    if (file == NULL) {
        printf("misc.c error: No file?\n"); // error
        return NULL;
    }

    fseek(file, 0, SEEK_END); // figuring out the size
    long size = ftell(file);
    rewind(file);
    *file_size = size;

    unsigned char *buffer = (unsigned char*) malloc((size + 1) * sizeof(unsigned char)); // create the buffer and fill it up with bytes
    fread(buffer, 1, size, file);
    buffer[size] = '\0';

    fclose(file);
    return buffer; // close the thing and return the thing
}

void countPairs(int *tokens, long size, int counts[MAX_VOCAB][MAX_VOCAB]){ // counts the pairs
    int i;
    for (i = 0; i < size - 1; i++) { // loop through tokens array and stop at size - 1
        if (tokens[i] < MAX_VOCAB && tokens[i+1] < MAX_VOCAB)
            counts[tokens[i]][tokens[i+1]]++; // increment the count for this pair
    }
}

void findMostCommonPair(int counts[MAX_VOCAB][MAX_VOCAB], int *best_a, int *best_b){ // finds the most common pair
    int max_count = 0, i, j; // variables
    for (i = 0; i < MAX_VOCAB; i++){ // loops through i and j
        for (j = 0; j < MAX_VOCAB; j++){
            if (counts[i][j] > max_count){ // is it the most common pair?
                max_count = counts[i][j]; // update the max count
                *best_a = i; *best_b = j; // update the most common a and b
            }
        }
    }
}

int *mergePair(int *tokens, long size, int pair_a, int pair_b, int new_token_id, long *new_size){ // merges pairs (BPE in a nutshell)
    int *new_tokens = (int*) malloc(size * sizeof(int)); // new tokens!!!!!
    int j = 0, i; // creates index variable and loop vafiable
    for (i = 0; i < size; i++){ // loop
        if (i < size - 1 && tokens[i] == pair_a && tokens[i+1] == pair_b){ // if it found the Perfect Pair <3
            new_tokens[j] = new_token_id; // puts the new token id into new tokens
            j++; i++; // incrementations
        } else { // else
            new_tokens[j] = tokens[i]; // puts tokens into new tokens
            j++; // increments j
        }
    }
    *new_size = j; // sets new size to j
    return new_tokens; // returns the New and Merged pairs
}
void trainBPE(const char *filename, int target_vocab_size, int *merges_a, int *merges_b, int *num_merges) { // BPE train functiom
    if (target_vocab_size > MAX_VOCAB) { // error
        printf("misc.c error: target_vocab_size is too large!\n");
        return;
    }

    long file_size;
    unsigned char *bytes = readFileToBytes(filename, &file_size); // reads file to bytes
    if (bytes == NULL) return; // error

    int *current_tokens = (int*) malloc(file_size * sizeof(int)); // Convert unsigned char array to int array
    int i;
    for (i = 0; i < file_size; i++) { // loop
        current_tokens[i] = bytes[i];
    }
    free(bytes); // free the bytes
    long current_size = file_size;

    int vocab_size = 256;
    for (; vocab_size < target_vocab_size; vocab_size++) { // BPE Training Loop
        static int counts[MAX_VOCAB][MAX_VOCAB]; // use static so it goes on the heap instead of the stack (avoids [[PROGRAMMING HELP WEBSITE UNFORTUNATLY REPLACED BY AI]])
        memset(counts, 0, sizeof(counts)); // zero out the counts array

        countPairs(current_tokens, current_size, counts); // count pairs

        int best_a, best_b;
        findMostCommonPair(counts, &best_a, &best_b); // find best pair

        if (counts[best_a][best_b] == 0) {
            printf("No more pairs to merge!\n");
            break; // stop if no pairs left
        }

        printf("Merging %d and %d into %d (count: %d)\n", best_a, best_b, vocab_size, counts[best_a][best_b]);
        merges_a[vocab_size - 256] = best_a;
        merges_b[vocab_size - 256] = best_b;

        long new_size;
        int *new_tokens = mergePair(current_tokens, current_size, best_a, best_b, vocab_size, &new_size); // merge

        free(current_tokens); // free old tokens
        current_tokens = new_tokens; // update pointers
        current_size = new_size;
    }

    free(current_tokens); // free final tokens
    *num_merges = vocab_size - 256;
}

void saveBPE(const char *filename, int *merges_a, int *merges_b, int num_merges){ // save tokenizer function
    FILE *f = fopen(filename, "wb"); // opens a New File(TM)
    if (f == NULL) { // error
        printf("misc.c error: saveBPE was given a NULL file\n");
        return;
    }
    fwrite(&num_merges, sizeof(int), 1, f); // writes
    fwrite(merges_a, sizeof(int), num_merges, f);
    fwrite(merges_b, sizeof(int), num_merges, f);
    fclose(f); // closes the file
}

void loadBPE(const char *filename, int *merges_a, int *merges_b, int *num_merges){
    FILE *f = fopen(filename, "rb"); // opens the BPE file
    if (f == NULL) { // error
        printf("misc.c error: loadBPE was given a NULL file\n");
        return;
    }
    fread(num_merges, sizeof(int), 1, f); // reads
    fread(merges_a, sizeof(int), *num_merges, f);
    fread(merges_b, sizeof(int), *num_merges, f);
    fclose(f); // closes the file
}

int *tokenize(const char *text, long *out_size, int *merges_a, int *merges_b, int num_merges){ // tokenize
    long text_length = strlen(text); // text length
    int *current_tokens = (int*) malloc(text_length * sizeof(int)); // Allocates current tokens array

    for (int i = 0; i < text_length; i++) { // loops through the text
        current_tokens[i] = (unsigned char)text[i]; // puts current tokens in array
    }

    long current_size = text_length; // set the current size to the text length

    for (int i = 0; i < num_merges; i++){ // loop through merges
        long new_size; // im so sleepy
        int *new_tokens = mergePair(current_tokens, current_size, merges_a[i], merges_b[i], 256 + i, &new_size); // merges

        free(current_tokens); // free the tokens

        current_tokens = new_tokens; // sets current tokens to new tokens
        current_size = new_size; // sets current size to new size
        *out_size = current_size; // sets out size to current size
    }
    return current_tokens; // returns the tokens
}

void expandToken(int token_id, int *merges_a, int *merges_b, int num_merges, unsigned char *buffer, int *buf_idx) { // expands token id into original bytes
    if (token_id < 256) { // raw byte, just add it to the buffer
        buffer[(*buf_idx)++] = (unsigned char)token_id;
        return;
    }

    int merge_idx = token_id - 256; //merged token, just split it back into its two parents
    if (merge_idx < num_merges) {
        expandToken(merges_a[merge_idx], merges_a, merges_b, num_merges, buffer, buf_idx);
        expandToken(merges_b[merge_idx], merges_a, merges_b, num_merges, buffer, buf_idx);
    }
}

/* ------- *\
   | I/O |
\* ------- */

void saveTransformerModel(const char *filename, embedding *embed, transformer_network *net, layer *unembed){ // saving transformer model function
    FILE *f = fopen(filename, "wb"); // opens file
    if (f == NULL) { // error
        printf("misc.c error: Couldn't open file in saveTransformerModel\n");
        return;
    }

    fwrite(embed->weights->data, sizeof(double), embed->weights->rows * embed->weights->columns, f); // writes embed weights

    fwrite(unembed->weights->data, sizeof(double), unembed->weights->rows * unembed->weights->columns, f); // writes unembed weights and biases
    fwrite(unembed->biases->data, sizeof(double), unembed->biases->rows * unembed->biases->columns, f);

    for (int i = 0; i < net->num_blocks; i++) {
        fwrite(net->blocks[i]->wq.weights->data, sizeof(double), net->blocks[i]->wq.weights->rows * net->blocks[i]->wq.weights->columns, f); // saves wq, wk, wv, wo and feedforward weights and biases
        fwrite(net->blocks[i]->wq.biases->data, sizeof(double), net->blocks[i]->wq.biases->rows * net->blocks[i]->wq.biases->columns, f);

        fwrite(net->blocks[i]->wk.weights->data, sizeof(double), net->blocks[i]->wk.weights->rows * net->blocks[i]->wk.weights->columns, f);
        fwrite(net->blocks[i]->wk.biases->data, sizeof(double), net->blocks[i]->wk.biases->rows * net->blocks[i]->wk.biases->columns, f);

        fwrite(net->blocks[i]->wv.weights->data, sizeof(double), net->blocks[i]->wv.weights->rows * net->blocks[i]->wv.weights->columns, f);
        fwrite(net->blocks[i]->wv.biases->data, sizeof(double), net->blocks[i]->wv.biases->rows * net->blocks[i]->wv.biases->columns, f);

        fwrite(net->blocks[i]->wo.weights->data, sizeof(double), net->blocks[i]->wo.weights->rows * net->blocks[i]->wo.weights->columns, f);
        fwrite(net->blocks[i]->wo.biases->data, sizeof(double), net->blocks[i]->wo.biases->rows * net->blocks[i]->wo.biases->columns, f);

        fwrite(net->blocks[i]->feedforward.weights->data, sizeof(double), net->blocks[i]->feedforward.weights->rows * net->blocks[i]->feedforward.weights->columns, f);
        fwrite(net->blocks[i]->feedforward.biases->data, sizeof(double), net->blocks[i]->feedforward.biases->rows * net->blocks[i]->feedforward.biases->columns, f);
    }

    fclose(f); // closes the file
}

void loadTransformerModel(const char *filename, embedding *embed, transformer_network *net, layer *unembed){ // loading transformer model function
    FILE *f = fopen(filename, "rb"); // opens file
    if (f == NULL) { // error
        printf("misc.c error: Couldn't open file in loadTransformerModel\n");
        return;
    }

    fread(embed->weights->data, sizeof(double), embed->weights->rows * embed->weights->columns, f); // reads embed weights

    fread(unembed->weights->data, sizeof(double), unembed->weights->rows * unembed->weights->columns, f); // reads unembed weights and biases
    fread(unembed->biases->data, sizeof(double), unembed->biases->rows * unembed->biases->columns, f);

    for (int i = 0; i < net->num_blocks; i++) {
        fread(net->blocks[i]->wq.weights->data, sizeof(double), net->blocks[i]->wq.weights->rows * net->blocks[i]->wq.weights->columns, f); // reads wq, wk, wv, wo and feedforward weights and biases
        fread(net->blocks[i]->wq.biases->data, sizeof(double), net->blocks[i]->wq.biases->rows * net->blocks[i]->wq.biases->columns, f);

        fread(net->blocks[i]->wk.weights->data, sizeof(double), net->blocks[i]->wk.weights->rows * net->blocks[i]->wk.weights->columns, f);
        fread(net->blocks[i]->wk.biases->data, sizeof(double), net->blocks[i]->wk.biases->rows * net->blocks[i]->wk.biases->columns, f);

        fread(net->blocks[i]->wv.weights->data, sizeof(double), net->blocks[i]->wv.weights->rows * net->blocks[i]->wv.weights->columns, f);
        fread(net->blocks[i]->wv.biases->data, sizeof(double), net->blocks[i]->wv.biases->rows * net->blocks[i]->wv.biases->columns, f);

        fread(net->blocks[i]->wo.weights->data, sizeof(double), net->blocks[i]->wo.weights->rows * net->blocks[i]->wo.weights->columns, f);
        fread(net->blocks[i]->wo.biases->data, sizeof(double), net->blocks[i]->wo.biases->rows * net->blocks[i]->wo.biases->columns, f);

        fread(net->blocks[i]->feedforward.weights->data, sizeof(double), net->blocks[i]->feedforward.weights->rows * net->blocks[i]->feedforward.weights->columns, f);
        fread(net->blocks[i]->feedforward.biases->data, sizeof(double), net->blocks[i]->feedforward.biases->rows * net->blocks[i]->feedforward.biases->columns, f);
    }

    fclose(f); // closes the file
}
