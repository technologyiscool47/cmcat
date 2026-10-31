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

#include <stdio.h> // From the ashes of fallen kingdoms, the First Dragon forged the Runes of C. We are but mortals wielding their power. Now speak your incantations and see what stirs.
#include <stdlib.h> // same but for stdlib.h
#include <math.h> // BOOOORINGGGGGG
#include <stdint.h> // for bitnet optimizations
#include <string.h> // for memset
#include <omp.h> // for multi threading

#define LN_EPSILON 1e-5f // super small number. is used later on
#define NUM_KV_GROUPS 2 // kv groups
#define MAX_CONTEXT 256 // Maximum tokens the kv cache can hold
#define NUM_HEADS 8 // Spiders usually have 4 pairs of eyes, having 8 eyes in total. This line declares 8 heads for multi head attention. Attention heads are like eyes.
                    // Does that mean that cmcat is a spider? Let's find out! Usinfg advanced analysis, it has become clear that me, technologyiscool47 is scared of
                    // spiders. However, I am not scared of cmcat which means that spiders and this project are not the same thing.


// WARNING: At the time i'm writing this, i am a complete beginner to C. i apologise for the comments



/* ---------------- *\
   | Arena stuffs |
\* ---------------- */



#define ARENA_CHUNK_SIZE (16 * 1024 * 1024) // 16 MB per chunk

typedef struct ArenaChunk { // arena chunk struct
    unsigned char *buffer; // buffer
    size_t offset; // offset
    size_t size; // size
    struct ArenaChunk *next; // next
} ArenaChunk; // Alias

ArenaChunk *arena_head = NULL; // sets the arena head and arena current to NULL
ArenaChunk *arena_current = NULL;

void initArena() { // function to initialize an arena
    arena_head = malloc(sizeof(ArenaChunk)); // allocates an arena chunk
    arena_head->buffer = malloc(ARENA_CHUNK_SIZE); // allocates a buffer
    arena_head->offset = 0; // sets the offset to 0
    arena_head->size = ARENA_CHUNK_SIZE; // sets the size
    arena_head->next = NULL; // null
    arena_current = arena_head; // sets the current arena
} /*                                                                                                                                                                                                */

void *arenaAlloc(size_t size) { // allocating arena function
    size = (size + 7) & ~((size_t)7); // Align to 8 bytes for floats

    if (arena_current->offset + size > arena_current->size) { // if the current chunk doesn't have enough space, allocate a new one
        size_t new_chunk_size = ARENA_CHUNK_SIZE; // sets the new chunk size
        if (size > new_chunk_size) new_chunk_size = size; // Handle massive matrices

        ArenaChunk *new_chunk = malloc(sizeof(ArenaChunk)); // allocates a new chunk
        new_chunk->buffer = malloc(new_chunk_size); // allocates the buffer
        new_chunk->offset = 0; // sets the offset to 0
        new_chunk->size = new_chunk_size; // sets the size
        new_chunk->next = NULL; // null

        arena_current->next = new_chunk; // sets the next to the new chunk
        arena_current = new_chunk; // sets the current chunk to the new chunk
    }

    void *ptr = &arena_current->buffer[arena_current->offset]; // pointer
    arena_current->offset += size; // blah blah blah
    return ptr; // returns
}

void resetArena() { // resets the arena
    ArenaChunk *c = arena_head; // sets c to the arena head
    while (c != NULL) { // while c exists
        c->offset = 0; // reset the offset, but keep the buffer allocated
        c = c->next; // sets c to c next
    }
    arena_current = arena_head; // Start back at the first chunk
}


/* --------------------------- *\
   | Matrix stuff and things |
\* --------------------------- */



typedef struct{ // create a struct named matrix
    int rows; // self-explanatory
    int columns; // self-explanatory
    float *data; // create a pointer to float named data
} matrix; // ending of the typedef phrase thing, name

matrix *newMatrix(int rows, int columns) { // create a function (the code i used as a reference used the matrix struct instead of int for some reason)
    if (rows <= 0 || columns <= 0) { printf("cmai.c error: asked to create matrix with negative rows or columns"); return NULL; } // if the rows or columns are 0, return NULL

    matrix *m = arenaAlloc(sizeof(matrix)); // create a matrix variable called m, which is also a pointer. then gets the size of a matrix struct and allocates it to the arena
    if (!m) { printf("cmai.c error: allocation of matrix failed"); return NULL; }

    m->rows = rows; // set the m pointer's rows to the function's rows
    m->columns = columns; // same but with columns
    m->data = arenaAlloc(rows * columns * sizeof(float)); // allocates the m pointer's data
    if (!m->data) { // checks if the allocation succeeded
        free(m); // frees m
        printf("cmai.c error: allocation of matrix data failed");
        return NULL; // returns null
    }

    memset(m->data, 0, rows * columns * sizeof(float)); // zeroes out memory

    return m;
}

void freeMatrix(matrix *m) { (void)m; } // freematrix used to free a matrix but with the arena it doesnt matter anymore. i am keeping the function to avoid deleting it everywhere

matrix *newPersistentMatrix(int rows, int columns) { // persistent matrix making function. takes rows and columns
    if (rows <= 0 || columns <= 0) { printf("cmai.c error: asked to create persistent matrix with negative rows or columns"); return NULL; } // if the rows and columns are less than 0 return null

    matrix *m = malloc(sizeof(matrix)); // allocates
    if (!m) { printf("cmai.c error: allocation of persistent matrix failed"); return NULL; } // if allocation failed, return null and warning

    m->rows = rows; // sets rows
    m->columns = columns; // sets columns
    m->data = malloc(rows * columns * sizeof(float)); // allocates data
    if (!m->data) { // if allocation failed
        free(m); // free
        printf("cmai.c error: allocation of persistent matrix data failed"); // error
        return NULL; // return
    }

    memset(m->data, 0, rows * columns * sizeof(float)); // zeroes out data

    return m; // returns
}

void freePersistentMatrix(matrix *m) { // frees persistent matrix
    if (m == NULL) { printf("cmai.c error: asked to free matrix that doesn't exist"); return; } // if m is null return
    free(m->data); // frees
    free(m); // frees
}

float getVal(matrix *m, int row, int col) { // makes a function called getval, takes a matrix pointer, rows value and columns value
    if (row < 0 || row >= m->rows || col < 0 || col >= m->columns) return NAN; // bounds check
    if (m == NULL) return NAN; // m being null check
    return m->data[row * m->columns + col]; // returns
}

void setVal(matrix *m, int row, int col, float val) { // maeks a function called setVal, its very similar to getval but it also takes a float value that we're going to set
    if (row < 0 || row >= m->rows || col < 0 || col >= m->columns) return; // bounds check
    m->data[row * m->columns + col] = val; // sets the data at the specified rows and columns to val
} // yes

void printMatrix(matrix *m) { // makes a function called printMatrix, takes a matrix pointer
    int i, j; // declare variables for loops
    for (i = 0; i < m->rows; i++) { // loop for rows
        for (j = 0; j < m->columns; j++) { // loop for columns
            printf("%.2f ", getVal(m, i, j)); // gets the value and prints it
        }
        printf("\n"); // prints a new line every time we have a new row
    }
} //

matrix *addMatrix(matrix *a, matrix *b) { // creates a function called addMatrix, takes 2 matrix pointers
    if (a->rows != b->rows || a->columns != b->columns) return NULL; // if a's rows arent the same as b's rows or the same with columns, return null
    matrix *result = newMatrix(a->rows, a->columns); // make a matrix called result, with a's rows and columns

    int total = a->rows * a->columns; // total
    // #pragma omp parallel for schedule(static) // super very important!!!! instructions for the compiler to multi thread the following loop
    for (int i = 0; i < total; i++) // iterates through every element in the matrix
        result->data[i] = a->data[i] + b->data[i]; // makes result's data the sum of a and b's data

    return result; // returns the result
}

matrix *transposeMatrix(matrix *m) { // makes a function called transposeMatrix, takes a matrix pointer
    matrix *result = newMatrix(m->columns, m->rows); // makes a matrix called result, takes makes m's columns its rows and likewise for m's rows

    #pragma omp parallel for schedule(static) // instruction for compiler to run this on multiple threads
    for (int i = 0; i < m->rows; i++) { // loop
        for (int j = 0; j < m->columns; j++) { // loop
            result->data[j * m->rows + i] = m->data[i * m->columns + j]; // sets the value at j,i to the value at i, j
        }
    }
    return result; // returns the result
}

matrix *multiplyMatrix(matrix *a, matrix *b) { // takes 2 matrices
    if (a->columns != b->rows) return NULL; // if a's columns aren't b's rows, return null
    matrix *result = newMatrix(a->rows, b->columns); // makes a result matrix

    #pragma omp parallel for schedule(static) // mega important multi threading instruction
    for (int i = 0; i < a->rows; i++) { // loops
        for (int k = 0; k < a->columns; k++) { // loops
            float a_val = a->data[i * a->columns + k]; // math
            for (int j = 0; j < b->columns; j++) {
                result->data[i * b->columns + j] += a_val * b->data[k * b->columns + j];
            }
        }
    }
    return result; // returns the result
}

float linear(float x) { // linear activation, literally does nothing
    return x; // does nothing
}

float relu(float x) { // makes a function
    if (x < 0) return 0; // If x is less than 0, return 0
    return x; // return x
}

float sigmoid(float x) { // makes the sigmoid function
    return 1.0f / (1.0f + expf(-x)); // return one difided by the sum of one plus the exponent of negative x
}

float silu(float x) { // silu
    return x * sigmoid(x); // returns x multiplied by sigmoid of x
}

float silu_derivative(float x) { // derivative of silu
    float s = sigmoid(x); // sets s to the sigmoid of x
    return s + x * s * (1.0f - s); // math
}

matrix *multiplyElementWise(matrix *a, matrix *b) { // Element multiplication
    matrix *result = newMatrix(a->rows, a->columns); // sets result to a matrix
    int total = a->rows * a->columns; // total elements in a
    for (int i = 0; i < total; i++) { // loops through everything
        result->data[i] = a->data[i] * b->data[i]; // multiplies the data
    }
    return result; // returns the result
}

int get2BitValue(float weight, float threshold) { // Maps a ternary value to a 2 bit binary value
    if (weight > threshold) return 2;  // 1  -> 10
    if (weight < -threshold) return 0; // -1 -> 00
    return 1;                          // 0  -> 01
}

void quantizeAndPack(matrix *float_weights, uint8_t **packed_weights, float *scale) {
    int total_weights = float_weights->rows * float_weights->columns;
    float abs_sum = 0.0f;
    for (int i = 0; i < total_weights; i++) { // Find the absolute maximum to use as our scale
        abs_sum += fabsf(float_weights->data[i]); // set the value to be the maximum
    }
    *scale = abs_sum/total_weights; // scale

    float threshold = 0.5 * (*scale); // BitNet uses 40% of absmax

    int packed_size = (total_weights + 3) / 4;

    if (*packed_weights != NULL) {
        free(*packed_weights); // Prevent memory leak if called twice!
    }

    *packed_weights = malloc(packed_size * sizeof(uint8_t)); // Allocate the packed array (1 byte per 4 weights)


    for (int i = 0; i < packed_size; i++) { // pack the weights 4 at a time
        uint8_t byte = 0;

        for (int j = 0; j < 4; j++) { // Process the 4 weights in the byte
            int weight_idx = i * 4 + j;

            if (weight_idx < total_weights) { // if we run out of weights just pad with 0s (isn't it better to pad with that one value we're not gonna use in the weight?)
                float w = float_weights->data[weight_idx];
                int two_bit = get2BitValue(w, threshold);

                int shift = 6 - (j * 2); // shift it to the correct land
                byte |= (two_bit << shift); // or it into the byte
            }
        }

        (*packed_weights)[i] = byte; // put the packed weights into the byte
    }
}

typedef struct { // adam optimizer. sgd (was used previously) is bad so i gotta use this
    matrix *m; // First moment
    matrix *v; // Second moment
    int t;     // Timestep
} adam_state; // Alias

adam_state *createAdamState(matrix *weights) { // creates an adam state opti,izer thing
    adam_state *state = malloc(sizeof(adam_state)); // allocates
    state->m = newPersistentMatrix(weights->rows, weights->columns); // creates matrices
    state->v = newPersistentMatrix(weights->rows, weights->columns);
    state->t = 0; // sets timestep to 0
    return state; // returns state
}

void adamUpdate(matrix *weights, matrix *grads, adam_state *state, float lr) { // updates adam
    state->t++; // increments timestep
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f; //  beta 1, beta 2 and epsilon but different

    float bias1 = 1.0f - powf(beta1, state->t);
    float bias2 = 1.0f - powf(beta2, state->t);

    for (int i = 0; i < weights->rows * weights->columns; i++) { // adam update of mystery. powf no more
        float g = grads->data[i];
        state->m->data[i] = beta1 * state->m->data[i] + (1.0f - beta1) * g; // math
        state->v->data[i] = beta2 * state->v->data[i] + (1.0f - beta2) * g * g;

        float m_hat = state->m->data[i] / bias1;
        float v_hat = state->v->data[i] / bias2;

        weights->data[i] -= lr * m_hat / (sqrtf(v_hat) + eps); // updates
    }
}

typedef struct {
    matrix *K; // Persistent cache for K
    matrix *V; // Persistent cache for V
    int len; // How many tokens are currently in the cache
} kv_cache;


/* --------------------------------------------- *\
   | Layers and other stuff on top of matrices |
\* --------------------------------------------- */


// < From here, the math gets more complicated, i copypasted some of it, i mostly know what it does but i guess it works >

typedef struct { // new struct
    matrix *weights; // master float weights (only used for Adam updates)
    uint8_t *packed_weights; // bitnet weights
    float scale; // scaling factor for the whole layer
    matrix *unpacked_weights; // unpacked weights because optimization
    matrix *biases; // biases
    float (*activation)(float); // activation
    adam_state *weight_adam; // adam weight
    adam_state *bias_adam; // adam bias
} layer;

matrix *unpackWeights(uint8_t *packed_weights, float scale, int rows, int cols) { // unpacks packed weights
    matrix *float_weights = newPersistentMatrix(rows, cols); // float weights
    int total = rows * cols; // total

    for (int i = 0; i < total; i++) { // loops
        int byte_idx = i / 4; // unpacks weights
        int bit_offset = 6 - ((i % 4) * 2);
        uint8_t two_bit = (packed_weights[byte_idx] >> bit_offset) & 0x03;

        float val = 0.0f; // sets the value
        if (two_bit == 2) val = 1.0f;
        else if (two_bit == 0) val = -1.0f;

        float_weights->data[i] = val * scale; // sets the float weights
    }
    return float_weights;// returns the float weights
}

matrix *ternaryMultiplyMatrix(matrix *input, uint8_t *packed_weights, float scale, int weight_rows, int weight_cols) { // matrix multiplication but ternary and fast and very cmcat
    matrix *result = newMatrix(input->rows, weight_cols); // result matrix
    #pragma omp parallel for schedule(static) // multi threading
    for (int i = 0; i < input->rows; i++) { // loops
        for (int k = 0; k < input->columns; k++) { // loops
            float input_val = input->data[i * input->columns + k]; // math of mystery, evil and esotericy

            for (int j = 0; j < weight_cols; j++) {
                int weight_idx = k * weight_cols + j;
                int byte_idx = weight_idx >> 2;
                int bit_offset = 6 - ((weight_idx & 3) << 1);
                uint8_t two_bit = (packed_weights[byte_idx] >> bit_offset) & 0x03;

                if (two_bit == 2) result->data[i * weight_cols + j] += input_val;
                else if (two_bit == 0) result->data[i * weight_cols + j] -= input_val;
            }
        }
    }
    int total = input->rows * weight_cols;
    for (int i = 0; i < total; i++) result->data[i] *= scale;
    return result; // returns
}

matrix *forward(layer *l, matrix *input, matrix *preact_out) { // i HATE forward passes
    matrix *weighted = ternaryMultiplyMatrix(input, l->packed_weights, l->scale, l->weights->rows, l->weights->columns); // sleep with one eye open
    int i, j; // variables
    for (i = 0; i < weighted->rows; i++) // disgusting loop
        for (j = 0; j < weighted->columns; j++) { // disgusting loop
            float val = getVal(weighted, i, j) + getVal(l->biases, 0, j); // evil declaration of a variable, adds biases
            setVal(preact_out, i, j, val); // evil preact variable. part of a horrible fix later on
            val = l->activation(val); // horrible variable
            setVal(weighted, i, j, val); // sets a value
        }
    return weighted; // returs
}

float mse(matrix *predicted, matrix *actual) { // takes two matrices
    if (predicted->rows != actual->rows || predicted->columns != actual->columns) return -1; // if the rows and columns aren't the same across the two matirices, return -1
    float sum = 0; //
    int i; // variable
    for (i = 0; i < predicted->rows * predicted->columns; i++) { // loop
        float diff = predicted->data[i] - actual->data[i]; // does a buncha math
        sum += diff * diff; // very esoteric math
    }
    return sum / (predicted->rows * predicted->columns); // returns
}

float linear_derivative(float x) { // linear derivative, literally just returns 1
    return 1.0f; // the derivative of x is always 1
}

float relu_derivative(float x) { // relu but derivative :thumbsup:
    return x > 0 ? 1.0f : 0.0f; // math.
}

float sigmoid_derivative(float x) { // sigmoid but derivative
    float s = sigmoid(x); // sigmoid
    return s * (1.0f - s); // derivative
}

float activation_derivative(float (*func)(float), float x) { // absolutely, yes, of course
    if (func == linear) return linear_derivative(x);
    if (func == relu) return relu_derivative(x);
    if (func == sigmoid) return sigmoid_derivative(x);
    if (func == silu) return silu_derivative(x);
    return 0;
}

void clipGradients(matrix *m, float max_norm) { // gradient clipping to stop them from exploding // used to be right after transformerBackward but i have to use this in backward so the definition must be before backward
    float total_norm = 0.0f;
    int i;

    for (i = 0; i < m->rows * m->columns; i++) { // hklsfzhkfgzyl
        if (isnan(m->data[i]) || isinf(m->data[i])) { // if we encounter infinity or a nan, reset gradient to 0
            for (i = 0; i < m->rows * m->columns; i++) m->data[i] = 0.0f;
            return;
        }
        total_norm += m->data[i] * m->data[i];
    }
    total_norm = sqrtf(total_norm);

    if (total_norm > max_norm) { // if it's too big, shrink the damn gradients
        float scale = max_norm / total_norm;
        for (i = 0; i < m->rows * m->columns; i++) {
            m->data[i] *= scale;
        }
    }
}

matrix *backward(layer *l, matrix *input, matrix *output_grad, float learning_rate, matrix *preact) { // backprop is hard.
    // output_grad is the gradient from the loss function
    // we need to propagate it back and update weights/biases

    matrix *activation_grad = newMatrix(output_grad->rows, output_grad->columns); // declaration of independence
    int i, j; // declaration of variables

    // apply activation derivative to output_grad
    for (i = 0; i < output_grad->rows; i++) { // loop
        for (j = 0; j < output_grad->columns; j++) { // loop
            float preact_val = getVal(preact, i, j); // preact stuff required for an annoying fix
            float act_deriv = activation_derivative(l->activation, preact_val);
            float grad_val = getVal(output_grad, i, j);
            setVal(activation_grad, i, j, grad_val * act_deriv); // sets a value
        }
    }

    // compute weight gradients: input^T * activation_grad
    matrix *input_t = transposeMatrix(input); // computes weight gradients
    matrix *weight_grad = multiplyMatrix(input_t, activation_grad); // computes weight gradients

    // update weights
    adamUpdate(l->weights, weight_grad, l->weight_adam, learning_rate);

    // Update biases
    matrix *bias_grad = newMatrix(1, activation_grad->columns); // (we need a temporary matrix for the bias gradient)
    for (j = 0; j < activation_grad->columns; j++) {
        float bg = 0;
        for (i = 0; i < activation_grad->rows; i++) {
            bg += getVal(activation_grad, i, j);
        }
        setVal(bias_grad, 0, j, bg);
    }
    adamUpdate(l->biases, bias_grad, l->bias_adam, learning_rate); // updates
    freeMatrix(bias_grad); // frees

    int total = l->weights->rows * l->weights->columns; // unpacks the weights in place because optimization
    for (int i = 0; i < total; i++) {
        int byte_idx = i / 4;
        int bit_offset = 6 - ((i % 4) * 2);
        uint8_t two_bit = (l->packed_weights[byte_idx] >> bit_offset) & 0x03;

        float val = 0.0f;
        if (two_bit == 2) val = 1.0f;
        else if (two_bit == 0) val = -1.0f;

        l->unpacked_weights->data[i] = val * l->scale;
    }

    matrix *weights_t = transposeMatrix(l->unpacked_weights); // Compute input gradient using cache
    matrix *input_grad = multiplyMatrix(activation_grad, weights_t);

    float abs_sum = 0.0f; // Re quantize in place (super mega faster than previously)
    for (int i = 0; i < total; i++) {
        abs_sum += fabsf(l->weights->data[i]);
    }
    l->scale = abs_sum / total;
    float threshold = 0.5f * l->scale;
    int packed_size = (total + 3) / 4;

    for (int i = 0; i < packed_size; i++) {
        uint8_t byte = 0;
        for (int j = 0; j < 4; j++) {
            int weight_idx = i * 4 + j;
            if (weight_idx < total) {
                float w = l->weights->data[weight_idx];
                int two_bit = get2BitValue(w, threshold);
                int shift = 6 - (j * 2);
                byte |= (two_bit << shift); // comment
            }
        }
        l->packed_weights[i] = byte; // overwrite in place
    }

    freeMatrix(weights_t); // Free me from this hell
    freeMatrix(activation_grad);
    freeMatrix(input_t);
    freeMatrix(weight_grad);

    return input_grad; // returns
}

matrix *layerNorm(matrix *m) { // new function. normalizes a matrix. named LAYERnorm to distinguish from another thing called batch normalization, which is older and worse.
                               // i repeat, it takes a MATRIX, not a layer. do not confuse the two. it is named layer normalization because it is not batch normalization
                               // do not confuse yourself

    matrix *out = newMatrix(m->rows, m->columns); // output matrix
    int i, j; // loop variables!!!
    float mean, variance, std_dev; // more variables

    for (i = 0; i < m->rows; i++) { // loopy loop loop!!! loops through rows
        mean = 0; // sets mean to 0

        for (j = 0; j < m->columns; j++) { // loops through columns
            mean += m->data[i * m->columns + j]; // adds together mean and m's data at i rows and j columns
        }
        mean /= m->columns; // divides mean by m's columns

        variance = 0; // variance time
        for (j = 0; j < m->columns; j++) { // loops through columns
            float diff = m->data[i * m->columns + j] - mean; // sets diff to m's data at i and j
            variance += diff * diff; // math
        }
        variance /= m->columns; // divides variance by columns

        std_dev = sqrtf(variance + LN_EPSILON); // sets std_dev to the square root of variance and epsilon (small number)
        for (j = 0; j < m->columns; j++) { // another loop through columns
            float normalized = (m->data[i * m->columns + j] - mean) / std_dev; // sets normalized to m's data subtracted by mean divided by std_dev
            setVal(out, i, j, normalized); // sets out to normalized
        }
    }
    return out; // returns
}

matrix *rmsNorm(matrix *m) { // rmsnorm. much newer and faster than layernorm and does the same thing
    matrix *out = newMatrix(m->rows, m->columns); // output matrix
    int i, j; // loop variables

    for (i = 0; i < m->rows; i++) {
        float ms = 0.0f;// Calculate the mean of squares
        for (j = 0; j < m->columns; j++) {
            float val = m->data[i * m->columns + j];
            ms += val * val;
        }
        ms /= m->columns;

        float rms = sqrtf(ms + LN_EPSILON); // calculate the RMS (add epsilon for safety)

        for (j = 0; j < m->columns; j++) { // Normalize
            out->data[i * m->columns + j] = m->data[i * m->columns + j] / rms;
        }
    }
    return out; // returns
}

/* ------------ *\
   | Networks |
\* ------------ */


//  </ From here, the math gets more complicated, i copypasted some of it, i mostly understand what it does but i guess it works >

typedef struct { // YAY! New struct
    layer *layers; // layers
    int num_layers; // how many layers
} network; // Elías

network *createNetwork(int *sizes, int num_layers, float (*activation)(float)) { // i have a good feeling about this
    if (num_layers <= 0) return NULL; // HELL YEAH WE'RE CHECKING IF THE num_layers IS LESS THAN 0
    network *net = malloc(sizeof(network)); // LETS GOOO we're allocating!!!!
    net->num_layers = num_layers; // YAHAHAHA THE num_layers IS num_layers!
    net->layers = malloc(num_layers * sizeof(layer)); // ALLOCATION TIMEEE

    int i, j; // YOO WE GOT LOOP VARIABLES
    for (i = 0; i < num_layers - 1; i++) { // LOOPY LOOPING LOOPS
        net->layers[i].weights = newMatrix(sizes[i], sizes[i+1]); // We're doing weights baby :D
        net->layers[i].biases = newMatrix(1, sizes[i+1]); // WOO HOO biases now
        net->layers[i].activation = activation; // Can't forget about the activation

        for (j = 0; j < sizes[i] * sizes[i+1]; j++) // MORE LOOPS HAHA
            net->layers[i].weights->data[j] = (rand() % 100) / 100.0f - 0.5; // random initialization for weights
    }

    return net; // We're returning!!!!! :)
}

// Alright the sugar rush ended

void freeNetwork(network *net) { // Make a function
    if (net == NULL) return; // If the network is NULL, return
    int i; // Variable
    for (i = 0; i < net->num_layers - 1; i++) { // Loop
        freeMatrix(net->layers[i].weights); // Two allocations, two frees
        freeMatrix(net->layers[i].biases); // mhm
    }
    free(net->layers); // Two allocations, two frees
    free(net); // mhm
}

typedef struct { // New struct just for the forward cache
    matrix **layer_outputs; // outputs from each layer
    matrix **layer_preacts; // new!!!!
    int num_outputs; // Number of outputs
} forward_cache; // Alias

void freeForwardCache(forward_cache *cache) { // Free forward cache
    if (cache == NULL) return; // if cache is null, return
    int i; // int i
    for (i = 0; i < cache->num_outputs - 1; i++) { // loop
        freeMatrix(cache->layer_outputs[i]); // free da matrix
        freeMatrix(cache->layer_preacts[i]); // free da other ANNOYING matrix
    }
    free(cache->layer_outputs); // free the layer outputs
    free(cache->layer_preacts); // i hate preacts so i free them
    free(cache); // free the cache
}

forward_cache *networkForward(network *net, matrix *input) { // makes a function that takes a network and an input
    forward_cache *cache = malloc(sizeof(forward_cache)); // allocating the cache
    cache->num_outputs = net->num_layers; // making the num_outputs num_layers
    cache->layer_outputs = malloc(net->num_layers * sizeof(matrix *)); // allocating the outputs
    cache->layer_preacts = malloc(net->num_layers * sizeof(matrix *)); // allocating the ANNOYING preacts

    matrix *current = input; // making the current matrix  the input
    int i; // i

    for (i = 0; i < net->num_layers - 1; i++) { // loopy loop loop
        cache->layer_preacts[i] = newMatrix(current->rows, net->layers[i].weights->columns);
        matrix *output = forward(&net->layers[i], current, cache->layer_preacts[i]); // sets the output to the forward pass of something
        cache->layer_outputs[i] = output; // sets the caches layer outputs to output
        current = output; // sets current to output
    }

    cache->layer_outputs[net->num_layers - 1] = input; // store input for backprop

    return cache; // mystery
}

void networkBackward(network *net, forward_cache *cache, matrix *output_grad, float learning_rate) { // backward pass function for networks. takes:
    // a network, a forward cache, a matrix which will be the output and a learning rate

    matrix *current_grad = output_grad; // makes a matrix called current_grad, sets it to output_grad
    int i; // loop variable

    for (i = net->num_layers - 2; i >= 0; i--) { // loop
        matrix *layer_input = (i == 0) ? cache->layer_outputs[net->num_layers - 1] : cache->layer_outputs[i - 1]; // makes a matrix called layer input, does a whole buncha stuff i dont understand :)
        matrix *prev_grad = backward(&net->layers[i], layer_input, current_grad, learning_rate, cache->layer_preacts[i]); // makes yet another matrix called prev_grad, sets it to a backward pass of something

        if (current_grad != output_grad) freeMatrix(current_grad); // if the current gradient isnt the same as output gradient, free current_grad
        current_grad = prev_grad; // sets current_grad to prev_grad
    }

    if (current_grad != output_grad) freeMatrix(current_grad); // if the current gradient isnt the same as output gradient, free current_grad
}


/* ----------------- *\
   | Miscellaneous |
\* ----------------- */


void saveNetwork(network *net, const char *filename) { // save Network
    FILE *f = fopen(filename, "wb"); // open file
    if (f == NULL) return; // if it failed, return

    int i; // mysterious
    for (i = 0; i < net->num_layers - 1; i++) { // loop
        fwrite(net->layers[i].weights->data, sizeof(float), net->layers[i].weights->rows * net->layers[i].weights->columns, f); // write net->layers[i].weights->data to f. the fwrite function is very difficult :(((
        fwrite(net->layers[i].biases->data, sizeof(float), net->layers[i].biases->rows * net->layers[i].biases->columns, f); // same but with biases
    }

    fclose(f); // close file
}

void loadNetwork(network *net, const char *filename) { // load Network
    FILE *f = fopen(filename, "rb"); // open file
    if (f == NULL) return; // if it failed, quit

    int i; // i dont know what this does
    for (i = 0; i < net->num_layers - 1; i++) { // loop
        fread(net->layers[i].weights->data, sizeof(float), net->layers[i].weights->rows * net->layers[i].weights->columns, f); // similar to the fwrites in saveNetwork, also difficult :(
        fread(net->layers[i].biases->data, sizeof(float), net->layers[i].biases->rows * net->layers[i].biases->columns, f); // biases
    }

    fclose(f); // CLOSE CLOSE CLOSE
}

/* --------------------------- *\
   | LLM / Transformer stuff |
\* --------------------------- */

void softmax(matrix *m) { // soft max function. takes a matrix
    int i, j; // loop variables

    for (i = 0; i < m->rows; i++) { // loop for the rows
        float max_val = m->data[i * m->columns]; // find max in this row
        for (j = 0; j < m->columns; j++) { // yet another loop
            if (m->data[i * m->columns + j] > max_val) // very complicated copypasted code
                max_val = m->data[i * m->columns + j]; // same
        }

        float sum = 0; // sum of exponentials
        for (j = 0; j < m->columns; j++) { // loopy loop loop
            float val = m->data[i * m->columns + j] - max_val; // subtract max for stability because everyone does it
            m->data[i * m->columns + j] = expf(val); // e^x
            sum += m->data[i * m->columns + j]; // sum
        }


        for (j = 0; j < m->columns; j++) { // loop
            m->data[i * m->columns + j] /= sum; // normalize so they sum to 1
        }
    }
}

void applyRoPE(matrix *x, int seq_len, int d_model, int is_backward, int start_pos) { // apply rotary positional encoding
    int head_dim = d_model / NUM_HEADS; // head dimensions
    int half = head_dim / 2; // half of head dimensions

    float freqs[half];
    for (int i = 0; i < half; i++)
        freqs[i] = 1.0f / powf(10000.0f, (2.0f * i) / head_dim);

    for (int m = 0; m < seq_len; m++) { // loops
        int current_pos = m + start_pos; // absolute position
        for (int h = 0; h < NUM_HEADS; h++) { // loops
            int offset = h * head_dim; // sets the offset to loop 2 by head dimensiions

            for (int i = 0; i < half; i++) { // loops through the half
                float freq = freqs[i]; // The frequency for this specific dimension pair
                float angle = current_pos * freq; // angle

                // In the backward pass, rotate by the inverse angle
                if (is_backward) angle = -angle; // if this is use in backward invert angle

                float cos_val = cosf(angle);
                float sin_val = sinf(angle);

                int idx1 = m * d_model + offset + i; // index 1
                int idx2 = m * d_model + offset + i + half; // index 1 + half

                float v1 = x->data[idx1]; // this
                float v2 = x->data[idx2]; // that

                // Apply the 2D rotation
                x->data[idx1] = v1 * cos_val - v2 * sin_val; // magic
                x->data[idx2] = v1 * sin_val + v2 * cos_val; // Should i try smoking Chinese medicinal herbs to increase my luck stat?
            }
        }
    }
}

matrix *attention(matrix *queries, matrix *keys, matrix *values) { // Multi-Head attention!
    int seq_len = queries->rows;
    int d_model = queries->columns;
    int head_dim = d_model / NUM_HEADS; // dimensions per head
    int kv_dim = head_dim * NUM_KV_GROUPS; // kv dimensions

    applyRoPE(queries, seq_len, d_model, 0, 0);
    applyRoPE(keys, seq_len, kv_dim, 0, 0); // kv

    matrix *final_output = newMatrix(seq_len, d_model); // final output matrix

    int h, i, j, k;
    for (h = 0; h < NUM_HEADS; h++) { // process each head independently
        // Map the query heads to the kv groups
        int kv_group = h / (NUM_HEADS / NUM_KV_GROUPS);
        int q_offset = h * head_dim;
        int kv_offset = kv_group * head_dim;

        matrix *scores = newMatrix(seq_len, seq_len);
        float scale = 1.0f / sqrtf(head_dim); // Scale by sqrt head dim

        for (i = 0; i < seq_len; i++) { // loop
            for (j = 0; j < seq_len; j++) { // loop
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    // Only multiply the [[HEAD DIM]] columns belonging to this head
                    float q_val = getVal(queries, i, q_offset + d);
                    float k_val = getVal(keys, j, kv_offset + d); // Using kv_offset
                    dot += q_val * k_val;
                }
                setVal(scores, i, j, dot * scale);
            }
        }

        // Causal Mask
        for (i = 0; i < seq_len; i++) {
            for (j = 0; j < seq_len; j++) {
                if (j > i) setVal(scores, i, j, -10000.0f);
            }
        }

        softmax(scores); // softmax over the scores

        for (i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (j = 0; j < seq_len; j++) {
                    val += getVal(scores, i, j) * getVal(values, j, kv_offset + d); // Using kv_offset
                }
                setVal(final_output, i, q_offset + d, val); // Glue heads output into final matrix
            }
        }
        freeMatrix(scores);
    }
    return final_output; // returns
}

typedef struct { // new struct!!!!! we lowkey have to save some values in transformerForward that were previously (if you want to see what i mean, go into a previous commit. we love git) being freed for simplicity
    matrix *input; // self explanatory
    matrix *norm1; // same
    matrix *Q; // same
    matrix *K; // same
    matrix *V; // same
    matrix *preact_q; // same
    matrix *preact_k; // same
    matrix *preact_v; // same
    matrix *preact_o; // same
    matrix *preact_ff; // same
    matrix *attn_out; // same
    matrix *after_attn; // same
    matrix *norm2; // same
    matrix *ff_out; // same
    matrix *output; // same
    matrix *preact_gate; // SwiGLU
    matrix *gate;
    matrix *preact_up;
    matrix *up;
    matrix *prod; // The result of gate * up
} transformer_cache; // alias

transformer_cache *createTransformerCache() { // create transformer cache function
    transformer_cache *cache = malloc(sizeof(transformer_cache)); // allocates a transformer cache
    if (cache == NULL) return NULL; // error check
    return cache; // returns
}

void freeTransformerCache(transformer_cache *cache) {
    if (cache == NULL) return;
    freeMatrix(cache->norm1);
    freeMatrix(cache->Q);
    freeMatrix(cache->K);
    freeMatrix(cache->V);
    freeMatrix(cache->preact_q);
    freeMatrix(cache->preact_k);
    freeMatrix(cache->preact_v);
    freeMatrix(cache->attn_out);
    freeMatrix(cache->after_attn);
    freeMatrix(cache->norm2);

    // Free the SwiGLU cache matrices
    freeMatrix(cache->preact_gate);
    freeMatrix(cache->gate);
    freeMatrix(cache->preact_up);
    freeMatrix(cache->up);
    freeMatrix(cache->prod);
    freeMatrix(cache->preact_ff);
    freeMatrix(cache->ff_out);

    freeMatrix(cache->output);
    free(cache);
}

typedef struct { // We're doing transformers now
    layer wq; // Query projection
    layer wk; // Key projection
    layer wv; // Value projection
    layer wo; // Output projection
    layer ff_w_gate; // SwiGLU things
    layer ff_w_up;
    layer ff_w_down;
} transformer_block; // Alias

transformer_block *createTransformerBlock(int input_size, int hidden_size, float (*activation)(float)) { // creates a transformer block. takes an input size, a hidden size and an activation
    transformer_block *block = calloc(1, sizeof(transformer_block)); // allocates the block
    if (block == NULL) return NULL;

    int kv_dim = (input_size / NUM_HEADS) * NUM_KV_GROUPS; // kv dimensions

    block->wq.weights = newPersistentMatrix(input_size, input_size); // projection initialization. its very repetitive so no comment
    block->wq.biases = newPersistentMatrix(1, input_size);
    block->wq.activation = linear;

    block->wk.weights = newPersistentMatrix(input_size, kv_dim);
    block->wk.biases = newPersistentMatrix(1, kv_dim);
    block->wk.activation = linear;

    block->wv.weights = newPersistentMatrix(input_size, kv_dim);
    block->wv.biases = newPersistentMatrix(1, kv_dim);
    block->wv.activation = linear;

    block->wo.weights = newPersistentMatrix(input_size, input_size);
    block->wo.biases = newPersistentMatrix(1, input_size);
    block->wo.activation = linear;

    block->wq.weight_adam = createAdamState(block->wq.weights); // adam state stuffs
    block->wq.bias_adam = createAdamState(block->wq.biases);

    block->wk.weight_adam = createAdamState(block->wk.weights);
    block->wk.bias_adam = createAdamState(block->wk.biases);

    block->wv.weight_adam = createAdamState(block->wv.weights);
    block->wv.bias_adam = createAdamState(block->wv.biases);

    block->wo.weight_adam = createAdamState(block->wo.weights);
    block->wo.bias_adam = createAdamState(block->wo.biases);

    // Randomize the projection weights
    float scale = 1.0f / sqrtf(input_size);

    int i;
    // Randomize wq and wo
    for (i = 0; i < block->wq.weights->rows * block->wq.weights->columns; i++) {
        block->wq.weights->data[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale;
        block->wo.weights->data[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale;
    }

    // Randomize wk and wv
    for (i = 0; i < block->wk.weights->rows * block->wk.weights->columns; i++) {
        block->wk.weights->data[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale;
        block->wv.weights->data[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale;
    }

    quantizeAndPack(block->wq.weights, &block->wq.packed_weights, &block->wq.scale); // quantize loss
    block->wq.unpacked_weights = unpackWeights(block->wq.packed_weights, block->wq.scale, block->wq.weights->rows, block->wq.weights->columns); // unpack into cache
    quantizeAndPack(block->wk.weights, &block->wk.packed_weights, &block->wk.scale);
    block->wk.unpacked_weights = unpackWeights(block->wk.packed_weights, block->wk.scale, block->wk.weights->rows, block->wk.weights->columns);
    quantizeAndPack(block->wv.weights, &block->wv.packed_weights, &block->wv.scale);
    block->wv.unpacked_weights = unpackWeights(block->wv.packed_weights, block->wv.scale, block->wv.weights->rows, block->wv.weights->columns);
    quantizeAndPack(block->wo.weights, &block->wo.packed_weights, &block->wo.scale);
    block->wo.unpacked_weights = unpackWeights(block->wo.packed_weights, block->wo.scale, block->wo.weights->rows, block->wo.weights->columns);

    // Swiglu


    block->ff_w_gate.weights = newPersistentMatrix(input_size, hidden_size); // Gate layer (input_size -> hidden_size)
    block->ff_w_gate.biases = newPersistentMatrix(1, hidden_size);
    block->ff_w_gate.activation = linear; // SiLU is applied manually later
    block->ff_w_gate.weight_adam = createAdamState(block->ff_w_gate.weights);
    block->ff_w_gate.bias_adam = createAdamState(block->ff_w_gate.biases);
    float scale_gate = 1.0f / sqrtf(input_size);
    for (int i = 0; i < input_size * hidden_size; i++) {
        block->ff_w_gate.weights->data[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale_gate;
    }
    quantizeAndPack(block->ff_w_gate.weights, &block->ff_w_gate.packed_weights, &block->ff_w_gate.scale);
    block->ff_w_gate.unpacked_weights = unpackWeights(block->ff_w_gate.packed_weights, block->ff_w_gate.scale, block->ff_w_gate.weights->rows, block->ff_w_gate.weights->columns);

    block->ff_w_up.weights = newPersistentMatrix(input_size, hidden_size); // Up layer (input_size -> hidden_size)
    block->ff_w_up.biases = newPersistentMatrix(1, hidden_size);
    block->ff_w_up.activation = linear;
    block->ff_w_up.weight_adam = createAdamState(block->ff_w_up.weights);
    block->ff_w_up.bias_adam = createAdamState(block->ff_w_up.biases);
    float scale_up = 1.0f / sqrtf(input_size);
    for (int i = 0; i < input_size * hidden_size; i++) {
        block->ff_w_up.weights->data[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale_up;
    }
    quantizeAndPack(block->ff_w_up.weights, &block->ff_w_up.packed_weights, &block->ff_w_up.scale);
    block->ff_w_up.unpacked_weights = unpackWeights(block->ff_w_up.packed_weights, block->ff_w_up.scale, block->ff_w_up.weights->rows, block->ff_w_up.weights->columns);

    block->ff_w_down.weights = newPersistentMatrix(hidden_size, input_size); // Down layer (hidden_size -> input_size)
    block->ff_w_down.biases = newPersistentMatrix(1, input_size);
    block->ff_w_down.activation = linear;
    block->ff_w_down.weight_adam = createAdamState(block->ff_w_down.weights);
    block->ff_w_down.bias_adam = createAdamState(block->ff_w_down.biases);
    float scale_down = 1.0f / sqrtf(hidden_size);
    for (int i = 0; i < hidden_size * input_size; i++) {
        block->ff_w_down.weights->data[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale_down;
    }
    quantizeAndPack(block->ff_w_down.weights, &block->ff_w_down.packed_weights, &block->ff_w_down.scale);
    block->ff_w_down.unpacked_weights = unpackWeights(block->ff_w_down.packed_weights, block->ff_w_down.scale, block->ff_w_down.weights->rows, block->ff_w_down.weights->columns);

    return block; // returns the meaning of life
}

void freeLayer(layer *l) { // function to free a layer
    if (l == NULL) return; // if layer doesnt exist, return
    freePersistentMatrix(l->weights); // free
    freePersistentMatrix(l->biases);
    freePersistentMatrix(l->unpacked_weights);

    if (l->packed_weights != NULL) {
        free(l->packed_weights); // Free the 2-bit packed array
    }

    if (l->weight_adam != NULL) { // Free Adam states
        freePersistentMatrix(l->weight_adam->m);
        freePersistentMatrix(l->weight_adam->v);
        free(l->weight_adam);
    }
    if (l->bias_adam != NULL) {
        freePersistentMatrix(l->bias_adam->m);
        freePersistentMatrix(l->bias_adam->v);
        free(l->bias_adam);
    }
}

void freeTransformerBlock(transformer_block *block) {
    if (block == NULL) return; // if block doesnt exist, return

    // Free all the projection layers
    freeLayer(&block->wq);
    freeLayer(&block->wk);
    freeLayer(&block->wv);
    freeLayer(&block->wo);

    // Free the SwiGLU layers
    freeLayer(&block->ff_w_gate);
    freeLayer(&block->ff_w_up);
    freeLayer(&block->ff_w_down);

    free(block);
}

matrix *transformerForward(matrix *input, transformer_block *block, transformer_cache *cache) { // forward pass (which i HATE) on a transformer blocks. transformer caches make me sad
    cache->input = input; // saving input for backprop

    cache->norm1 = rmsNorm(input); // attention OUTput

    cache->preact_q = newMatrix(cache->norm1->rows, block->wq.weights->columns); // making new matrices for the projections
    cache->preact_k = newMatrix(cache->norm1->rows, block->wk.weights->columns);
    cache->preact_v = newMatrix(cache->norm1->rows, block->wv.weights->columns);
    cache->Q = forward(&block->wq, cache->norm1, cache->preact_q); // ughhh
    cache->K = forward(&block->wk, cache->norm1, cache->preact_k);
    cache->V = forward(&block->wv, cache->norm1, cache->preact_v);

    cache->attn_out = attention(cache->Q, cache->K, cache->V); // attention things

    cache->preact_o = newMatrix(cache->attn_out->rows, block->wo.weights->columns); // preact output
    matrix *proj_out = forward(&block->wo, cache->attn_out, cache->preact_o); // projection output

    cache->after_attn = addMatrix(input, proj_out); // after attention

    freeMatrix(proj_out);

    cache->norm2 = rmsNorm(cache->after_attn); // norm 2

    cache->preact_gate = newMatrix(cache->norm2->rows, block->ff_w_gate.weights->columns); // SwiGLU forward
    cache->gate = forward(&block->ff_w_gate, cache->norm2, cache->preact_gate);

    for (int i = 0; i < cache->gate->rows * cache->gate->columns; i++) {
        cache->gate->data[i] = silu(cache->gate->data[i]); // Apply SiLU to the gate in place
    }

    cache->preact_up = newMatrix(cache->norm2->rows, block->ff_w_up.weights->columns);
    cache->up = forward(&block->ff_w_up, cache->norm2, cache->preact_up);

    cache->prod = multiplyElementWise(cache->gate, cache->up);

    cache->preact_ff = newMatrix(cache->prod->rows, block->ff_w_down.weights->columns);
    cache->ff_out = forward(&block->ff_w_down, cache->prod, cache->preact_ff);

    cache->output = addMatrix(cache->after_attn, cache->ff_out);

    return cache->output; // returns the output
} // I want to make one thing clear: cmcat is not a learning project. It's two sevenths of a learning project at most, and even if it is, i am learning C. I am a complete poser in AI and i do not mind it.

typedef struct { // embedding struct
    matrix *weights; // giant lookup table for the embeddings
    int vocab_size; // vocab size
    int embedding_dim; // embedding dimensions
    adam_state *weight_adam; // adammmmm
} embedding; // alias

embedding *createEmbedding(int vocab_size, int embedding_dim) { // new function, makes an embedding, takes vocab size and the dimensions :thumbsup:
    embedding *e = malloc(sizeof(embedding)); // allocates
    if (e == NULL) return NULL; // error handling

    e->vocab_size = vocab_size; // sets vocab size
    e->embedding_dim = embedding_dim; // sets embedding dimensions
    e->weights = newPersistentMatrix(vocab_size, embedding_dim); // sets weights

    e->weight_adam = createAdamState(e->weights);

    float scale = 1.0f / sqrtf(embedding_dim);

    int i; // i
    for (i = 0; i < vocab_size * embedding_dim; i++) { // for loop
        e->weights->data[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * scale; // randomizes dimensions
    }

    return e; // returns e
}

void freeEmbedding(embedding *e) { // frees an embedding
    if (e == NULL) return; // error handling
    freePersistentMatrix(e->weights); // free the lookup table
    free(e); // free the struct
}

matrix *embeddingForward(embedding *e, int *token_ids, int seq_len) { // forward pass (I HATE THEM AJDAKASDKFHK) for an embedding
    matrix *out = newMatrix(seq_len, e->embedding_dim); // outout matrix

    int i, j; // loop variables
    for (i = 0; i < seq_len; i++) { // loops through the sequence length
        int token_id = token_ids[i]; // sets the token ids


        if (token_id < 0 || token_id >= e->vocab_size) { // Bounds check! don't look for a token that doesn't exist
            printf("token %d is out of bounds!", token_id); // error message
            continue;
        }

        for (j = 0; j < e->embedding_dim; j++) { // loop
            float val = getVal(e->weights, token_id, j); // gets a vector
            setVal(out, i, j, val); // puts it in an output
        }
    }

    return out; // returns
}

void embeddingBackward(embedding *e, int *token_ids, int seq_len, matrix *input_grad, float learning_rate) { // embedding backward with ADAM
    // Create a gradient matrix with the embedding weights size
    matrix *weight_grad = newMatrix(e->vocab_size, e->embedding_dim); // starts filled with 0s

    int i, j;
    for (i = 0; i < seq_len; i++) { // Populate the gradients for the tokens that were in sequence
        int token_id = token_ids[i];
        if (token_id < 0 || token_id >= e->vocab_size) continue;

        for (j = 0; j < e->embedding_dim; j++) {
            float grad = getVal(input_grad, i, j);
            float existing = getVal(weight_grad, token_id, j);
            setVal(weight_grad, token_id, j, existing + grad);
        }
    }

    adamUpdate(e->weights, weight_grad, e->weight_adam, learning_rate); // Update the embedding weights

    freeMatrix(weight_grad); // free the temporary matrix
}

matrix *unembeddingForward(matrix *input, layer *unembed) { // float unembedding forward pass
    matrix *logits = multiplyMatrix(input, unembed->weights); // logits
    // Add biases
    for (int i = 0; i < logits->rows; i++) {
        for (int j = 0; j < logits->columns; j++) {
            logits->data[i * logits->columns + j] += unembed->biases->data[j];
        }
    }
    softmax(logits); // softmax
    return logits; // returns
}

typedef struct { // new struct!!!! transformer networksssss
    transformer_block **blocks; // the transformer blocks
    int num_blocks; // how many blocks
} transformer_network; // alias

transformer_network *createTransformerNetwork(int num_blocks, int input_size, int hidden_size, float (*activation)(float)) { // new function!!! takes the number of transformer blocks, input size,
    if (num_blocks <= 0) return NULL; // if the number of blocks is less than 0, return null                                   // hidden size and an activation

    transformer_network *net = malloc(sizeof(transformer_network)); // allocates the network
    net->num_blocks = num_blocks; // sets number of blocks
    net->blocks = malloc(num_blocks * sizeof(transformer_block *)); // allocates blocks

    int i; // loop variable
    for (i = 0; i < num_blocks; i++) { // loops through number of blocks
        net->blocks[i] = createTransformerBlock(input_size, hidden_size, activation); // creates the blocks
    }

    return net; // returns the network
}

void freeTransformerNetwork(transformer_network *net) { // frees a transformer network
    if (net == NULL) return; // if the network is null, return
    int i; // loop variable
    for (i = 0; i < net->num_blocks; i++) { // loops through blocks
        freeTransformerBlock(net->blocks[i]); // frees blocks
    }
    free(net->blocks); // frees blocks Again(TM)
    free(net); // frees network
}

matrix *transformerNetworkForward(matrix *input, transformer_network *net, transformer_cache **caches) { // forward pass on a transformer network
    matrix *current = input; // sets current matrix to current matrix
    int i; // int i

    for (i = 0; i < net->num_blocks; i++) { // loops through blocks
        caches[i] = createTransformerCache(); // creates transformer caches for each block
        matrix *next = transformerForward(current, net->blocks[i], caches[i]); // forward pass REAL 100% FREE DOWNLOAD NO SMS NO EMAIL :thumbsup: DOWNLOAD NOW
        current = next; // sets current to next
    }

    return current; // returns current
}

float crossEntropyLoss(matrix *predicted, int *target_tokens) { // cross entropy loss. turns out the last one was incorrect or something idek im so tired
    float loss = 0;
    int i;

    for (i = 0; i < predicted->rows; i++) { // math
        int target = target_tokens[i];
        float prob = getVal(predicted, i, target);
        loss += -logf(prob + LN_EPSILON); // actual loss calculation
    }

    loss = loss / predicted->rows; // Average loss over all predictions
    return loss; //returns
}

void crossEntropyLossGradient(matrix *predicted, int *target_tokens, matrix *output_grad) { // cross entropy loss gradient thing. basically, when training, you have to use a combination of the loss
    int i, j; // loop variable                                                              // and softmax and it is very complicated, however, when you simplify the big scary equation you literally
    for (i = 0; i < predicted->rows; i++) { // rows loop                                    // get is taking your predicted probabilities, and subtract 1 from the correct answer
        int target = target_tokens[i]; // target

        for (j = 0; j < predicted->columns; j++) { // column loop
            float prob = getVal(predicted, i, j); // probablility

            if (j == target) // if ai got it right
                setVal(output_grad, i, j, (prob - 1.0f) / predicted->rows); // nudge it towards more confidence
            else // else
                setVal(output_grad, i, j, (prob - 0.0f) / predicted->rows); // nudge it towards the right answer
        }
    }
}

void attentionBackward(matrix *Q, matrix *K, matrix *V, matrix *attn_out_grad, matrix **Q_grad, matrix **K_grad, matrix **V_grad) {
    int seq_len = Q->rows;
    int d_model = Q->columns;
    int head_dim = d_model / NUM_HEADS;
    int kv_dim = head_dim * NUM_KV_GROUPS;

    matrix *q_g = newMatrix(seq_len, d_model);
    matrix *k_g = newMatrix(seq_len, kv_dim); // kv
    matrix *v_g = newMatrix(seq_len, kv_dim);

    int h, i, j;
    for (h = 0; h < NUM_HEADS; h++) {
        int kv_group = h / (NUM_HEADS / NUM_KV_GROUPS);
        int q_offset = h * head_dim;
        int kv_offset = kv_group * head_dim;

        // Recalculate softmax scores for this head
        matrix *S = newMatrix(seq_len, seq_len);
        float scale = 1.0f / sqrtf(head_dim);
        for (i = 0; i < seq_len; i++) {
            for (j = 0; j < seq_len; j++) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += getVal(Q, i, q_offset + d) * getVal(K, j, kv_offset + d);
                }
                setVal(S, i, j, dot * scale);
            }
        }
        // Apply causal mask
        for (i = 0; i < seq_len; i++) {
            for (j = 0; j < seq_len; j++) {
                if (j > i) setVal(S, i, j, -10000.0f);
            }
        }
        softmax(S);

        // V_grad (Accumulate into v_g)
        for (i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (j = 0; j < seq_len; j++) {
                    val += getVal(S, j, i) * getVal(attn_out_grad, j, q_offset + d);
                }
                float existing = getVal(v_g, i, kv_offset + d);
                setVal(v_g, i, kv_offset + d, existing + val); // accumulate
            }
        }

        // S_grad
        matrix *s_g = newMatrix(seq_len, seq_len);
        for (i = 0; i < seq_len; i++) {
            for (j = 0; j < seq_len; j++) {
                float val = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    val += getVal(attn_out_grad, i, q_offset + d) * getVal(V, j, kv_offset + d);
                }
                setVal(s_g, i, j, val);
            }
        }

        // Softmax backward
        for (i = 0; i < seq_len; i++) {
            float dot = 0.0f;
            for (j = 0; j < seq_len; j++) {
                dot += getVal(S, i, j) * getVal(s_g, i, j);
            }
            for (j = 0; j < seq_len; j++) {
                float s_val = getVal(S, i, j);
                float sg_val = getVal(s_g, i, j);
                setVal(s_g, i, j, s_val * (sg_val - dot));
            }
        }

        // Scale backward
        for (i = 0; i < seq_len * seq_len; i++) s_g->data[i] *= scale;

        // Q_grad (Accumulate into q_g)
        for (i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (j = 0; j < seq_len; j++) {
                    val += getVal(s_g, i, j) * getVal(K, j, kv_offset + d);
                }
                float existing = getVal(q_g, i, q_offset + d);
                setVal(q_g, i, q_offset + d, existing + val); // ACCUMULATE
            }
        }

        // K_grad (Accumulate into k_g)
        for (i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (j = 0; j < seq_len; j++) {
                    val += getVal(s_g, j, i) * getVal(Q, j, q_offset + d);
                }
                float existing = getVal(k_g, i, kv_offset + d);
                setVal(k_g, i, kv_offset + d, existing + val); // ACCUMULATE
            }
        }

        freeMatrix(S);
        freeMatrix(s_g);
    }

    applyRoPE(q_g, seq_len, d_model, 1, 0);
    applyRoPE(k_g, seq_len, kv_dim, 1, 0); // Keys are now kv_dim!

    *Q_grad = q_g;
    *K_grad = k_g;
    *V_grad = v_g;
}

matrix *layerNormBackward(matrix *grad_out, matrix *normalized_out) { // layer normalization backward
    matrix *grad_in = newMatrix(grad_out->rows, grad_out->columns); // esoteric matrix creation
    int i, j;
    int N = grad_out->columns; // number of features

    for (i = 0; i < grad_out->rows; i++) { // loop
        float sum_grad = 0.0f;
        float sum_grad_y = 0.0f; // variables

        for (j = 0; j < N; j++) { // sums
            float dy = getVal(grad_out, i, j);
            float y = getVal(normalized_out, i, j);
            sum_grad += dy;
            sum_grad_y += dy * y; // The moon is made of used coffee pucks
        }

        // Calculate final dx
        for (j = 0; j < N; j++) {
            float dy = getVal(grad_out, i, j);
            float y = getVal(normalized_out, i, j);
            float dx = (1.0f / N) * (dy - sum_grad / N - y * sum_grad_y / N);
            setVal(grad_in, i, j, dx);
        }
    }
    return grad_in; // yay
}

matrix *rmsNormBackward(matrix *grad_out, matrix *normalized_out, matrix *original_input) { // backward for rmsnorm
    matrix *grad_in = newMatrix(grad_out->rows, grad_out->columns); // grad in
    int i, j; // loop variablesdd
    int N = grad_out->columns; // N

    for (i = 0; i < grad_out->rows; i++) {
        float ms = 0.0f; // Calculate the RMS of the normalized output
        for (j = 0; j < N; j++) {
            float y = getVal(original_input, i, j);
            ms += y * y;
        }
        ms /= N;
        float rms_inv = 1.0f / sqrtf(ms + LN_EPSILON); // do not delete the epsilon

        float dot = 0.0f; // Calculate dot product of grad_out and normalized_out
        for (j = 0; j < N; j++) {
            dot += getVal(grad_out, i, j) * getVal(normalized_out, i, j);
        }

        for (j = 0; j < N; j++) { // 3. Final gradient
            float dy = getVal(grad_out, i, j);
            float y = getVal(normalized_out, i, j);
            float dx = rms_inv * (dy - y * dot / N);
            setVal(grad_in, i, j, dx);
        }
    }
    return grad_in; // returns
}

matrix *transformerBackward(transformer_block *block, transformer_cache *cache, matrix *output_grad, float learning_rate) { // transformer backward function. does a bunch of super esoteric math
    matrix *ff_out_grad = output_grad; // Backward through ff_w_down
    matrix *prod_grad = backward(&block->ff_w_down, cache->prod, ff_out_grad, learning_rate, cache->preact_ff);

    // Backward through elementwise multiply
    matrix *gate_grad = multiplyElementWise(prod_grad, cache->up); // prod = gate * up => d(gate) = d(prod) * up, d(up) = d(prod) * gate
    matrix *up_grad = multiplyElementWise(prod_grad, cache->gate);
    freeMatrix(prod_grad);

    for (int i = 0; i < gate_grad->rows * gate_grad->columns; i++) { // Backward through SiLU
        float preact_val = cache->preact_gate->data[i]; // gate = silu(preact_gate) => d(preact_gate) = d(gate) * silu_derivative(preact_gate)
        gate_grad->data[i] *= silu_derivative(preact_val);
    }

    matrix *norm2_grad_from_gate = backward(&block->ff_w_gate, cache->norm2, gate_grad, learning_rate, cache->preact_gate); // Backward through ff_w_gate and ff_w_up
    matrix *norm2_grad_from_up = backward(&block->ff_w_up, cache->norm2, up_grad, learning_rate, cache->preact_up);

    freeMatrix(gate_grad);
    freeMatrix(up_grad);

    matrix *norm2_out_grad = addMatrix(norm2_grad_from_gate, norm2_grad_from_up); // Sum the gradients because both came from norm2
    freeMatrix(norm2_grad_from_gate);
    freeMatrix(norm2_grad_from_up);

    matrix *after_attn_grad_from_ff = rmsNormBackward(norm2_out_grad, cache->norm2, cache->after_attn); // rmsnorm backward
    freeMatrix(norm2_out_grad); // Free the intermediate gradient

    matrix *after_attn_grad = addMatrix(output_grad, after_attn_grad_from_ff); // "residual split" for after attn
    freeMatrix(after_attn_grad_from_ff);

    matrix *proj_out_grad = after_attn_grad; // wo backward
    matrix *attn_out_grad = backward(&block->wo, cache->attn_out, proj_out_grad, learning_rate, cache->preact_o);

    matrix *Q_grad, *K_grad, *V_grad; // Attention backward
    attentionBackward(cache->Q, cache->K, cache->V, attn_out_grad, &Q_grad, &K_grad, &V_grad);
    freeMatrix(attn_out_grad);

    matrix *norm1_grad_q = backward(&block->wq, cache->norm1, Q_grad, learning_rate, cache->preact_q);// projections backward
    matrix *norm1_grad_k = backward(&block->wk, cache->norm1, K_grad, learning_rate, cache->preact_k);
    matrix *norm1_grad_v = backward(&block->wv, cache->norm1, V_grad, learning_rate, cache->preact_v);

    freeMatrix(Q_grad);
    freeMatrix(K_grad);
    freeMatrix(V_grad);

    matrix *norm1_grad_tmp = addMatrix(norm1_grad_q, norm1_grad_k); // Sum the gradients because Q, K, V all came from norm1
    matrix *norm1_grad = addMatrix(norm1_grad_tmp, norm1_grad_v);

    freeMatrix(norm1_grad_q);
    freeMatrix(norm1_grad_k);
    freeMatrix(norm1_grad_v);
    freeMatrix(norm1_grad_tmp);

    matrix *input_grad_part2 = rmsNormBackward(norm1_grad, cache->norm1, cache->input); // rmsnorm backward
    freeMatrix(norm1_grad);

    matrix *final_input_grad = addMatrix(after_attn_grad, input_grad_part2); // final residual addition

    freeMatrix(after_attn_grad); // Free me from this hell
    freeMatrix(input_grad_part2);

    return final_input_grad;
}

matrix *transformerForwardStep(matrix *input, transformer_block *block, kv_cache *kv) { // forward step
    int d_model = input->columns; // mere integers
    int kv_dim = block->wk.weights->columns;

    matrix *norm1 = rmsNorm(input); // RMSNorm

    matrix *preact_q = newMatrix(1, block->wq.weights->columns); // qkv projections
    matrix *Q = forward(&block->wq, norm1, preact_q);

    matrix *preact_k = newMatrix(1, kv_dim);
    matrix *K_new = forward(&block->wk, norm1, preact_k);

    matrix *preact_v = newMatrix(1, kv_dim);
    matrix *V_new = forward(&block->wv, norm1, preact_v);


    applyRoPE(Q, 1, d_model, 0, kv->len);// Apply rope
    applyRoPE(K_new, 1, kv_dim, 0, kv->len);


    for (int c = 0; c < kv_dim; c++) { // update kv cache
        kv->K->data[kv->len * kv_dim + c] = K_new->data[c];
        kv->V->data[kv->len * kv_dim + c] = V_new->data[c];
    }
    kv->len++;
    int cache_len = kv->len;

    int head_dim = d_model / NUM_HEADS; // Attention
    matrix *attn_out = newMatrix(1, d_model);
    float scale = 1.0f / sqrtf(head_dim);

    for (int h = 0; h < NUM_HEADS; h++) {
        int kv_group = h / (NUM_HEADS / NUM_KV_GROUPS);
        int q_offset = h * head_dim;
        int kv_offset = kv_group * head_dim;

        matrix *scores = newMatrix(1, cache_len);
        for (int j = 0; j < cache_len; j++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float q_val = getVal(Q, 0, q_offset + d);
                float k_val = getVal(kv->K, j, kv_offset + d);
                dot += q_val * k_val;
            }
            setVal(scores, 0, j, dot * scale);
        }

        softmax(scores); // softmax

        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int j = 0; j < cache_len; j++) {
                val += getVal(scores, 0, j) * getVal(kv->V, j, kv_offset + d);
            }
            setVal(attn_out, 0, q_offset + d, val);
        }
        freeMatrix(scores);
    }

    matrix *preact_o = newMatrix(1, block->wo.weights->columns); // output projection
    matrix *proj_out = forward(&block->wo, attn_out, preact_o);

    matrix *after_attn = addMatrix(input, proj_out); // residual add
    freeMatrix(proj_out);

    matrix *norm2 = rmsNorm(after_attn); // swiglu
    matrix *preact_gate = newMatrix(1, block->ff_w_gate.weights->columns);
    matrix *gate = forward(&block->ff_w_gate, norm2, preact_gate);
    for (int i = 0; i < gate->rows * gate->columns; i++) gate->data[i] = silu(gate->data[i]);

    matrix *preact_up = newMatrix(1, block->ff_w_up.weights->columns);
    matrix *up = forward(&block->ff_w_up, norm2, preact_up);

    matrix *prod = multiplyElementWise(gate, up);
    matrix *preact_ff = newMatrix(1, block->ff_w_down.weights->columns);
    matrix *ff_out = forward(&block->ff_w_down, prod, preact_ff);

    matrix *output = addMatrix(after_attn, ff_out); // 9. Final Residual

    // Cleanup
    freeMatrix(norm1);
    freeMatrix(preact_q); freeMatrix(Q);
    freeMatrix(preact_k); freeMatrix(K_new);
    freeMatrix(preact_v); freeMatrix(V_new);
    freeMatrix(attn_out);
    freeMatrix(preact_o);
    freeMatrix(after_attn);
    freeMatrix(norm2);
    freeMatrix(preact_gate); freeMatrix(gate);
    freeMatrix(preact_up); freeMatrix(up);
    freeMatrix(prod);
    freeMatrix(preact_ff); freeMatrix(ff_out);

    return output; // returns
}

matrix *transformerNetworkForwardStep(matrix *input, transformer_network *net, kv_cache **kv_caches) { /* 💀️ */
    matrix *current = input;
    for (int i = 0; i < net->num_blocks; i++) {
        current = transformerForwardStep(current, net->blocks[i], kv_caches[i]);
    }
    return current;
}
