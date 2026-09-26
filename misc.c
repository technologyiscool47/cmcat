#include <stdio.h>
#include <stdlib.h>
#include <string.h> // memset

#define MAX_VOCAB 10000 // max vocab token size

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
    for (i = 0; i < 256; i++){ // loops through i and j
        for (j = 0; j < 256; j++){
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
void trainBPE(const char *filename, int target_vocab_size) { // BPE train functiom
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

    for (int vocab_size = 256; vocab_size < target_vocab_size; vocab_size++) { // BPE Training Loop
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

        long new_size;
        int *new_tokens = mergePair(current_tokens, current_size, best_a, best_b, vocab_size, &new_size); // merge

        free(current_tokens); // free old tokens
        current_tokens = new_tokens; // update pointers
        current_size = new_size;
    }

    free(current_tokens); // free final tokens
}
