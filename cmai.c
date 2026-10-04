#include <stdio.h> // From the ashes of fallen kingdoms, the First Dragon forged the Runes of C. We are but mortals wielding their power. Now speak your incantations and see what stirs.
#include <stdlib.h> // same but for stdlib.h
#include <math.h> // BOOOORINGGGGGG

#define LN_EPSILON 1e-5 // super small number. is used later on
#define NUM_HEADS 8 // Spiders usually have 4 pairs of eyes, having 8 eyes in total. This line declares 8 heads for multi head attention. Attention heads are like eyes.
                    // Does that mean that cmcat is a spider? Let's find out! Using advanced analysis, it has become clear that me, technologyiscool47 is scared of
                    // spiders. However, I am not scared of cmcat which means that spiders and this project are not the same thing.


// WARNING: At the time i'm writing this, i am a complete beginner to C. i apologise for the comments



/* --------------------------- *\
   | Matrix stuff and things |
\* --------------------------- */



typedef struct{ // create a struct named matrix
    int rows; // self-explanatory
    int columns; // self-explanatory
    double *data; // create a pointer to double named data
} matrix; // ending of the typedef phrase thing, name

matrix *newMatrix(int rows, int columns) { // create a function (the code i used as a reference used the matrix struct instead of int for some reason)
    if (rows <= 0 || columns <= 0) return NULL; // if the rows or columns are 0, return NULL

    matrix *m = malloc(sizeof(matrix)); // create a matrix variable called m, which is also a pointer. then gets the size of a matrix struct and allocates it to the HEAP
    if (!m)
        return NULL;

    m->rows = rows; // set the m pointer's rows to the function's rows
    m->columns = columns; // same but with columns
    m->data = malloc(rows * columns * sizeof(double)); // allocates the m pointer's data
    if (!m->data) { // checks if the allocation succeeded
        free(m); // frees m
        return NULL; // returns null
    }

    int i; // creates a variable named i, for a loop

    for (i = 0; i < rows*columns; i++) // for loop
        m->data[i] = 0; // sets m's data to 0.

    return m;
}

void freeMatrix(matrix *m) { // makes a function named freeMatrix, it takes a pointer which has to be a matrix
    if (m == NULL) return; // if, for some reason, null is input here, we do not accept it.
    free(m->data); // frees m data
    free(m); // frees m
}

double getVal(matrix *m, int row, int col) { // makes a function called getval, takes a matrix pointer, rows value and columns value
    if (row < 0 || row >= m->rows || col < 0 || col >= m->columns) return NAN; // bounds check
    if (m == NULL) return NAN; // m being null check
    return m->data[row * m->columns + col]; // returns
}

void setVal(matrix *m, int row, int col, double val) { // maeks a function called setVal, its very similar to getval but it also takes a double value that we're going to set
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
    int i; // i don't know what this does. it's way too difficult for me to understand. someone please explain
    for (i = 0; i < a->rows * a->columns; i++) // iterates through every element in the matrix
        result->data[i] = a->data[i] + b->data[i]; // makes result's data the sum of a and b's data
    return result; // returns the result
}

matrix *transposeMatrix(matrix *m) { // makes a function called transposeMatrix, takes a matrix pointer
    matrix *result = newMatrix(m->columns, m->rows); // makes a matrix called result, takes makes m's columns its rows and likewise for m's rows
    int i, j; // makes 2 variables
    for (i = 0; i < m->rows; i++) // loop
        for (j = 0; j < m->columns; j++) // loop
            setVal(result, j, i, getVal(m, i, j)); // sets the value at j, i to the value at i, j
    return result; // returns the result
}

matrix *multiplyMatrix(matrix *a, matrix *b) { // takes two matrices
    if (a->columns != b->rows) return NULL; // if a's columns aren't b's rows, return null
    matrix *result = newMatrix(a->rows, b->columns); // makes a result matrix
    int i, j, k; // loop variables
    for (i = 0; i < a->rows; i++) // loop
        for (j = 0; j < b->columns; j++) // loop
            for (k = 0; k < a->columns; k++) // loop
                result->data[i * result->columns + j] += a->data[i * a->columns + k] * b->data[k * b->columns + j]; // dot product of mystery and evil
    return result; // returns the result
}

double linear(double x) { // linear activation, literally does nothing
    return x; // does nothing
}

double relu(double x) { // makes a function
    if (x < 0) return 0; // If x is less than 0, return 0
    return x; // return x
}

double sigmoid(double x) { // makes the sigmoid function
    return 1.0 / (1.0 + exp(-x)); // return one difided by the sum of one plus the exponent of negative x
}

typedef struct { // adam optimizer. sgd (was used previously) is bad so i gotta use this
    matrix *m; // First moment
    matrix *v; // Second moment
    int t;     // Timestep
} adam_state; // Alias

adam_state *createAdamState(matrix *weights) { // creates an adam state opti,izer thing
    adam_state *state = malloc(sizeof(adam_state)); // allocates
    state->m = newMatrix(weights->rows, weights->columns); // creates matrices
    state->v = newMatrix(weights->rows, weights->columns);
    state->t = 0; // sets timestep to 0
    return state; // returns state
}

void adamUpdate(matrix *weights, matrix *grads, adam_state *state, double lr) { // updates adam
    state->t++; // increments timestep
    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8; //  beta 1, beta 2 and epsilon but different

    for (int i = 0; i < weights->rows * weights->columns; i++) { // loop
        double g = grads->data[i]; // maths that i will not comment on
        state->m->data[i] = beta1 * state->m->data[i] + (1.0 - beta1) * g;
        state->v->data[i] = beta2 * state->v->data[i] + (1.0 - beta2) * g * g;

        double m_hat = state->m->data[i] / (1.0 - pow(beta1, state->t));
        double v_hat = state->v->data[i] / (1.0 - pow(beta2, state->t));

        weights->data[i] -= lr * m_hat / (sqrt(v_hat) + eps);
    }
}

/* --------------------------------------------- *\
   | Layers and other stuff on top of matrices |
\* --------------------------------------------- */


// < From here, the math gets more complicated, i copypasted some of it, i don't really know what it does but i guess it works >

typedef struct { // New struct!!! yay
    matrix *weights; // weights matrix
    matrix *biases; // biases matrix
    double (*activation)(double); // pointer to an activation function
    adam_state *weight_adam; // adam weights and biases
    adam_state *bias_adam;
} layer; // asdlkasdlkj

matrix *forward(layer *l, matrix *input, matrix *preact_out) { // i HATE forward passes
    matrix *weighted = multiplyMatrix(input, l->weights); // sleep with one eye open
    int i, j; // variables
    for (i = 0; i < weighted->rows; i++) // disgusting loop
        for (j = 0; j < weighted->columns; j++) { // disgusting loop
            double val = getVal(weighted, i, j) + getVal(l->biases, 0, j); // evil declaration of a variable, adds biases
            setVal(preact_out, i, j, val); // evil preact variable. part of a horrible fix later on
            val = l->activation(val); // horrible variable
            setVal(weighted, i, j, val); // sets a value
        }
    return weighted; // returs
}

double mse(matrix *predicted, matrix *actual) { // takes two matrices
    if (predicted->rows != actual->rows || predicted->columns != actual->columns) return -1; // if the rows and columns aren't the same across the two matirices, return -1
    double sum = 0; //
    int i; // variable
    for (i = 0; i < predicted->rows * predicted->columns; i++) { // loop
        double diff = predicted->data[i] - actual->data[i]; // does a buncha math
        sum += diff * diff; // very esoteric math
    }
    return sum / (predicted->rows * predicted->columns); // returns
}

double linear_derivative(double x) { // linear derivative, literally just returns 1
    return 1.0; // the derivative of x is always 1
}

double relu_derivative(double x) { // relu but derivative :thumbsup:
    return x > 0 ? 1.0 : 0.0; // math.
}

double sigmoid_derivative(double x) { // sigmoid but derivative
    double s = sigmoid(x); // sigmoid
    return s * (1.0 - s); // derivative
}

double activation_derivative(double (*func)(double), double x) { // absolutely, yes, of course
    if (func == linear) return linear_derivative(x); // ADD THIS
    if (func == relu) return relu_derivative(x);
    if (func == sigmoid) return sigmoid_derivative(x);
    return 0;
}

void clipGradients(matrix *m, double max_norm) { // gradient clipping to stop them from exploding // used to be right after transformerBackward but i have to use this in backward so the definition must be before backward
    double total_norm = 0.0;
    int i;

    for (i = 0; i < m->rows * m->columns; i++) { // hklsfzhkfgzyl
        if (isnan(m->data[i]) || isinf(m->data[i])) { // if we encounter infinity or a nan, reset gradient to 0
            for (i = 0; i < m->rows * m->columns; i++) m->data[i] = 0.0;
            return;
        }
        total_norm += m->data[i] * m->data[i];
    }
    total_norm = sqrt(total_norm);

    if (total_norm > max_norm) { // if it's too big, shrink the damn gradients
        double scale = max_norm / total_norm;
        for (i = 0; i < m->rows * m->columns; i++) {
            m->data[i] *= scale;
        }
    }
}

matrix *backward(layer *l, matrix *input, matrix *output_grad, double learning_rate, matrix *preact) { // backprop is hard.
    // output_grad is the gradient from the loss function
    // we need to propagate it back and update weights/biases

    matrix *activation_grad = newMatrix(output_grad->rows, output_grad->columns); // declaration of independence
    int i, j; // declaration of variables

    // apply activation derivative to output_grad
    for (i = 0; i < output_grad->rows; i++) { // loop
        for (j = 0; j < output_grad->columns; j++) { // loop
            double preact_val = getVal(preact, i, j); // preact stuff required for an annoying fix
            double act_deriv = activation_derivative(l->activation, preact_val);
            double grad_val = getVal(output_grad, i, j);
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
        double bg = 0;
        for (i = 0; i < activation_grad->rows; i++) {
            bg += getVal(activation_grad, i, j);
        }
        setVal(bias_grad, 0, j, bg);
    }
    adamUpdate(l->biases, bias_grad, l->bias_adam, learning_rate); // updates
    freeMatrix(bias_grad); // frees

    // compute input gradient: activation_grad * weights^T
    matrix *weights_t = transposeMatrix(l->weights);
    matrix *input_grad = multiplyMatrix(activation_grad, weights_t);

    freeMatrix(activation_grad);
    freeMatrix(input_t);
    freeMatrix(weight_grad);
    freeMatrix(weights_t); // Free me from this hell

    return input_grad;


}

matrix *layerNorm(matrix *m) { // new function. normalizes a matrix. named LAYERnorm to distinguish from another thing called batch normalization, which is older and worse.
                               // i repeat, it takes a MATRIX, not a layer. do not confuse the two. it is named layer normalization because it is not batch normalization
                               // do not confuse yourself

    matrix *out = newMatrix(m->rows, m->columns); // output matrix
    int i, j; // loop variables!!!
    double mean, variance, std_dev; // more variables

    for (i = 0; i < m->rows; i++) { // loopy loop loop!!! loops through rows
        mean = 0; // sets mean to 0

        for (j = 0; j < m->columns; j++) { // loops through columns
            mean += m->data[i * m->columns + j]; // adds together mean and m's data at i rows and j columns
        }
        mean /= m->columns; // divides mean by m's columns

        variance = 0; // variance time
        for (j = 0; j < m->columns; j++) { // loops through columns
            double diff = m->data[i * m->columns + j] - mean; // sets diff to m's data at i and j
            variance += diff * diff; // math
        }
        variance /= m->columns; // divides variance by columns

        std_dev = sqrt(variance + LN_EPSILON); // sets std_dev to the square root of variance and epsilon (small number)
        for (j = 0; j < m->columns; j++) { // another loop through columns
            double normalized = (m->data[i * m->columns + j] - mean) / std_dev; // sets normalized to m's data subtracted by mean divided by std_dev
            setVal(out, i, j, normalized); // sets out to normalized
        }
    }
    return out; // returns
}

/* ------------ *\
   | Networks |
\* ------------ */


//  </ From here, the math gets more complicated, i copypasted some of it, i don't really know what it does but i guess it works >

typedef struct { // YAY! New struct
    layer *layers; // layers
    int num_layers; // how many layers
} network; // Elías

network *createNetwork(int *sizes, int num_layers, double (*activation)(double)) { // i have a good feeling about this
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
            net->layers[i].weights->data[j] = (rand() % 100) / 100.0 - 0.5; // random initialization for weights
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

void networkBackward(network *net, forward_cache *cache, matrix *output_grad, double learning_rate) { // backward pass function for networks. takes:
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
        fwrite(net->layers[i].weights->data, sizeof(double), net->layers[i].weights->rows * net->layers[i].weights->columns, f); // write net->layers[i].weights->data to f. the fwrite function is very difficult :(((
        fwrite(net->layers[i].biases->data, sizeof(double), net->layers[i].biases->rows * net->layers[i].biases->columns, f); // same but with biases
    }

    fclose(f); // close file
}

void loadNetwork(network *net, const char *filename) { // load Network
    FILE *f = fopen(filename, "rb"); // open file
    if (f == NULL) return; // if it failed, quit

    int i; // i dont know what this does
    for (i = 0; i < net->num_layers - 1; i++) { // loop
        fread(net->layers[i].weights->data, sizeof(double), net->layers[i].weights->rows * net->layers[i].weights->columns, f); // similar to the fwrites in saveNetwork, also difficult :(
        fread(net->layers[i].biases->data, sizeof(double), net->layers[i].biases->rows * net->layers[i].biases->columns, f); // biases
    }

    fclose(f); // CLOSE CLOSE CLOSE
}

/* --------------------------- *\
   | LLM / Transformer stuff |
\* --------------------------- */

void softmax(matrix *m) { // soft max function. takes a matrix
    int i, j; // loop variables

    for (i = 0; i < m->rows; i++) { // loop for the rows
        double max_val = m->data[i * m->columns]; // find max in this row
        for (j = 0; j < m->columns; j++) { // yet another loop
            if (m->data[i * m->columns + j] > max_val) // very complicated copypasted code
                max_val = m->data[i * m->columns + j]; // same
        }

        double sum = 0; // sum of exponentials
        for (j = 0; j < m->columns; j++) { // loopy loop loop
            double val = m->data[i * m->columns + j] - max_val; // subtract max for stability because everyone does it
            m->data[i * m->columns + j] = exp(val); // e^x
            sum += m->data[i * m->columns + j]; // sum
        }


        for (j = 0; j < m->columns; j++) { // loop
            m->data[i * m->columns + j] /= sum; // normalize so they sum to 1
        }
    }
}

matrix *attention(matrix *queries, matrix *keys, matrix *values) { // Multi-Head attention!
    int seq_len = queries->rows;
    int d_model = queries->columns;
    int head_dim = d_model / NUM_HEADS; // dimensions per head

    matrix *final_output = newMatrix(seq_len, d_model); // This is where we glue all the heads back together at the end

    int h, i, j, k;

    for (h = 0; h < NUM_HEADS; h++) { // process each head independently
        int offset = h * head_dim; // which 8 columns are we looking at?

        // calculate scores for this head
        matrix *scores = newMatrix(seq_len, seq_len);
        double scale = 1.0 / sqrt(head_dim); // Scale by sqrt head dim

        for (i = 0; i < seq_len; i++) {
            for (j = 0; j < seq_len; j++) {
                double dot = 0.0;
                for (k = 0; k < head_dim; k++) {
                    // Only multiply the [[HEAD DIM]] columns belonging to this head
                    double q_val = getVal(queries, i, offset + k);
                    double k_val = getVal(keys, j, offset + k);
                    dot += q_val * k_val;
                }
                setVal(scores, i, j, dot * scale);
            }
        }

        // causal Mask
        for (i = 0; i < seq_len; i++) {
            for (j = 0; j < seq_len; j++) {
                if (j > i) {
                    setVal(scores, i, j, -10000.0);
                }
            }
        }

        softmax(scores); // softmax over the scores

        // Multiply scores by V_h to get the head's output
        for (i = 0; i < seq_len; i++) {
            for (k = 0; k < head_dim; k++) {
                double val = 0.0;
                for (j = 0; j < seq_len; j++) {
                    val += getVal(scores, i, j) * getVal(values, j, offset + k);
                }
                setVal(final_output, i, offset + k, val); // Glue heads output into final matrix
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
} transformer_cache; // alias

transformer_cache *createTransformerCache() { // create transformer cache function
    transformer_cache *cache = malloc(sizeof(transformer_cache)); // allocates a transformer cache
    if (cache == NULL) return NULL; // error check
    return cache; // returns
}

void freeTransformerCache(transformer_cache *cache) { // free transformer cache function
    if (cache == NULL) return; // checks if cache is null
    freeMatrix(cache->norm1); // free everything inside the transformer cache
    freeMatrix(cache->Q);
    freeMatrix(cache->K);
    freeMatrix(cache->V); // ok
    freeMatrix(cache->preact_q);
    freeMatrix(cache->preact_k);
    freeMatrix(cache->preact_v);
    freeMatrix(cache->attn_out);
    freeMatrix(cache->after_attn);
    freeMatrix(cache->norm2);
    freeMatrix(cache->ff_out);
    freeMatrix(cache->output);
    free(cache);
}

typedef struct { // We're doing transformers now
    layer wq; // Query projection
    layer wk; // Key projection
    layer wv; // Value projection
    layer wo; // Output projection
    layer feedforward; // layer
} transformer_block; // Alias

transformer_block *createTransformerBlock(int input_size, int hidden_size, double (*activation)(double)) { // creates a transformer block. takes an input size, a hidden size and an activation
    transformer_block *block = malloc(sizeof(transformer_block)); // allocates the block
    if (block == NULL) return NULL;

    block->wq.weights = newMatrix(input_size, input_size); // projection layer time. this is all just initialization, i wont comment on it
    block->wq.biases = newMatrix(1, input_size);
    block->wq.activation = linear; // projections usually use relu so we'll use that // EDIT: No they do not, im stupid and the models loss very rarely drops below 5 with relu
    block->wk.weights = newMatrix(input_size, input_size);
    block->wk.biases = newMatrix(1, input_size);
    block->wk.activation = linear;

    block->wv.weights = newMatrix(input_size, input_size);
    block->wv.biases = newMatrix(1, input_size);
    block->wv.activation = linear;

    block->wo.weights = newMatrix(input_size, input_size);
    block->wo.biases = newMatrix(1, input_size);
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
    double scale = 1.0 / sqrt(input_size);

    int i;
    for (i = 0; i < input_size * input_size; i++) {
        block->wq.weights->data[i] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * scale;
        block->wk.weights->data[i] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * scale;
        block->wv.weights->data[i] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * scale;
        block->wo.weights->data[i] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * scale;
    }

    block->feedforward.weights = newMatrix(input_size, input_size); // sets the weights to a matrix with rows(input) and columns(hidden size)
    block->feedforward.biases = newMatrix(1, input_size); // sets the biases to a matrix of 1 row and input_size columns
    block->feedforward.activation = activation; // sets the activation to activation

    block->feedforward.weight_adam = createAdamState(block->feedforward.weights); // more adam
    block->feedforward.bias_adam = createAdamState(block->feedforward.biases);

    for (i = 0; i < input_size * input_size; i++) // for loop
        block->feedforward.weights->data[i] = (rand() % 100) / 100.0 - 0.5; // sets the weights randomly

    return block; // returns the meaning of life
}

void freeTransformerBlock(transformer_block *block) { // function that frees the Transformer block
    if (block == NULL) return; // if there's no block, return
    freeMatrix(block->feedforward.weights); // frees
    freeMatrix(block->feedforward.biases);

    freeMatrix(block->wq.weights); // projection freeing. no comment
    freeMatrix(block->wq.biases);

    freeMatrix(block->wk.weights);
    freeMatrix(block->wk.biases);

    freeMatrix(block->wv.weights);
    freeMatrix(block->wv.biases);

    freeMatrix(block->wo.weights);
    freeMatrix(block->wo.biases);

    free(block);
}

matrix *transformerForward(matrix *input, transformer_block *block, transformer_cache *cache) { // forward pass (which i HATE) on a transformer blocks. transformer caches make me sad
    cache->input = input; // saving input for backprop

    cache->norm1 = layerNorm(input); // attention OUTput

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

    cache->norm2 = layerNorm(cache->after_attn); // norm 2
    cache->preact_ff = newMatrix(cache->norm2->rows, block->feedforward.biases->columns); // preact_ff matrix creation
    cache->ff_out = forward(&block->feedforward, cache->norm2, cache->preact_ff); // ff out forward pass thing

    cache->output = addMatrix(cache->after_attn, cache->ff_out);

    return cache->output; // returns the output
} // I want to make one thing clear: cmcat is not a learning project. It's two sevenths of a learning project at most, and even if it is, i am learning C. I am a complete poser in the world of AI and i do not mind it.

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
    e->weights = newMatrix(vocab_size, embedding_dim); // sets weights

    e->weight_adam = createAdamState(e->weights);

    double scale = 1.0 / sqrt(embedding_dim);

    int i; // i
    for (i = 0; i < vocab_size * embedding_dim; i++) { // for loop
        e->weights->data[i] = (((double)rand() / RAND_MAX) * 2.0 - 1.0) * scale; // randomizes dimensions
    }

    return e; // returns e
}

void freeEmbedding(embedding *e) { // frees an embedding
    if (e == NULL) return; // error handling
    freeMatrix(e->weights); // free the lookup table
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
            double val = getVal(e->weights, token_id, j); // gets a vector
            setVal(out, i, j, val); // puts it in an output
        }
    }

    return out; // returns
}

void embeddingBackward(embedding *e, int *token_ids, int seq_len, matrix *input_grad, double learning_rate) { // embedding backward with ADAM
    // Create a gradient matrix with the embedding weights size
    matrix *weight_grad = newMatrix(e->vocab_size, e->embedding_dim); // starts filled with 0s

    int i, j;
    for (i = 0; i < seq_len; i++) { // Populate the gradients for the tokens that were in sequence
        int token_id = token_ids[i];
        if (token_id < 0 || token_id >= e->vocab_size) continue;

        for (j = 0; j < e->embedding_dim; j++) {
            double grad = getVal(input_grad, i, j);
            setVal(weight_grad, token_id, j, grad);
        }
    }

    adamUpdate(e->weights, weight_grad, e->weight_adam, learning_rate); // Update the embedding weights

    freeMatrix(weight_grad); // free the temporary matrix
}

matrix *positionalEncoding(int seq_len, int d_model) { // new function. We're doing positional encoding ¡
    matrix *pe = newMatrix(seq_len, d_model); // makes a matrix called pe
    int i, j; // loop cariables

    for (i = 0; i < seq_len; i++) { // loops through rows
        for (j = 0; j < d_model; j++) { // loops through columns
            double freq = 1.0 / pow(10000.0, (2.0 * ((double)j / 2.0)) / d_model); // frequency variable

            if (j % 2 == 0) { // if j is even
                setVal(pe, i, j, sin(i * freq)); // sets sine
            } else { // else
                setVal(pe, i, j, cos(i * freq)); // sets cosine
            }
        }
    }
    return pe; // returns pe
}

matrix *unembeddingForward(matrix *input, layer *unembed) { // unembedding forward pass (which i HATE)
    matrix *preact = newMatrix(input->rows, unembed->weights->columns); // preact matrix
    matrix *logits = forward(unembed, input, preact); // logits matrix, forward pass

    freeMatrix(preact); // frees preact matrix
    softmax(logits); // softmaxes the logits

    return logits; // returns the logits
}

typedef struct { // new struct!!!! transformer networksssss
    transformer_block **blocks; // the transformer blocks
    int num_blocks; // how many blocks
} transformer_network; // alias

transformer_network *createTransformerNetwork(int num_blocks, int input_size, int hidden_size, double (*activation)(double)) { // new function!!! takes the number of transformer blocks, input size,
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

double crossEntropyLoss(matrix *predicted, int *target_tokens) { // cross entropy loss. turns out the last one was incorrect or something idek im so tired
    double loss = 0;
    int i;

    for (i = 0; i < predicted->rows; i++) { // math
        int target = target_tokens[i];
        double prob = getVal(predicted, i, target);
        loss += -log(prob + LN_EPSILON); // actual loss calculation
    }

    loss = loss / predicted->rows; // Average loss over all predictions
    return loss; //returns
}

void crossEntropyLossGradient(matrix *predicted, int *target_tokens, matrix *output_grad) { // cross entropy loss gradient thing. basically, when training, you have to use a combination of the loss
    int i, j; // loop variable                                                              // and softmax and it is very complicated, however, when you simplify the big scary equation you literally
    for (i = 0; i < predicted->rows; i++) { // rows loop                                    // get is taking your predicted probabilities, and subtract 1 from the correct answer
        int target = target_tokens[i]; // target

        for (j = 0; j < predicted->columns; j++) { // column loop
            double prob = getVal(predicted, i, j); // probablility

            if (j == target) // if ai got it right
                setVal(output_grad, i, j, (prob - 1.0) / predicted->rows); // nudge it towards more confidence
            else // else
                setVal(output_grad, i, j, (prob - 0.0) / predicted->rows); // nudge it towards the right answer
        }
    }
}

void attentionBackward(matrix *Q, matrix *K, matrix *V, matrix *attn_out_grad, matrix **Q_grad, matrix **K_grad, matrix **V_grad) { // backward for attention. rewritten because multi head
    int seq_len = Q->rows; // sets up variables
    int d_model = Q->columns;
    int head_dim = d_model / NUM_HEADS;

    matrix *q_g = newMatrix(seq_len, d_model); // sets up gradients
    matrix *k_g = newMatrix(seq_len, d_model);
    matrix *v_g = newMatrix(seq_len, d_model);

    int h, i, j, k;

    for (h = 0; h < NUM_HEADS; h++) {
        int offset = h * head_dim;

        // Recalculate softmax scores for this head
        matrix *S = newMatrix(seq_len, seq_len);
        double scale = 1.0 / sqrt(head_dim);
        for (i = 0; i < seq_len; i++) {
            for (j = 0; j < seq_len; j++) {
                double dot = 0.0;
                for (k = 0; k < head_dim; k++) {
                    dot += getVal(Q, i, offset + k) * getVal(K, j, offset + k);
                }
                setVal(S, i, j, dot * scale);
            }
        }
        // Apply causal mask
        for (i = 0; i < seq_len; i++) {
            for (j = 0; j < seq_len; j++) {
                if (j > i) setVal(S, i, j, -10000.0);
            }
        }
        softmax(S);

        // V_grad for this head
        for (i = 0; i < seq_len; i++) {
            for (k = 0; k < head_dim; k++) {
                double val = 0.0;
                for (j = 0; j < seq_len; j++) {
                    val += getVal(S, j, i) * getVal(attn_out_grad, j, offset + k);
                }
                setVal(v_g, i, offset + k, val);
            }
        }

        // S_grad
        matrix *s_g = newMatrix(seq_len, seq_len);
        for (i = 0; i < seq_len; i++) {
            for (j = 0; j < seq_len; j++) {
                double val = 0.0;
                for (k = 0; k < head_dim; k++) {
                    val += getVal(attn_out_grad, i, offset + k) * getVal(V, j, offset + k);
                }
                setVal(s_g, i, j, val);
            }
        }

        // Softmax backward
        for (i = 0; i < seq_len; i++) {
            double dot = 0.0;
            for (j = 0; j < seq_len; j++) {
                dot += getVal(S, i, j) * getVal(s_g, i, j);
            }
            for (j = 0; j < seq_len; j++) {
                double s_val = getVal(S, i, j);
                double sg_val = getVal(s_g, i, j);
                setVal(s_g, i, j, s_val * (sg_val - dot));
            }
        }

        // Scale backward
        for (i = 0; i < seq_len * seq_len; i++) s_g->data[i] *= scale;

        // Q_grad and K_grad
        for (i = 0; i < seq_len; i++) {
            for (k = 0; k < head_dim; k++) {
                double q_val = 0.0;
                double k_val = 0.0;
                for (j = 0; j < seq_len; j++) {
                    q_val += getVal(s_g, i, j) * getVal(K, j, offset + k);
                    k_val += getVal(s_g, j, i) * getVal(Q, i, offset + k);
                }
                setVal(q_g, i, offset + k, q_val);
                setVal(k_g, i, offset + k, k_val);
            }
        }

        freeMatrix(S); // frees
        freeMatrix(s_g);
    }

    *Q_grad = q_g; // sets the gradients
    *K_grad = k_g;
    *V_grad = v_g;
}

matrix *layerNormBackward(matrix *grad_out, matrix *normalized_out) { // layer normalization backward
    matrix *grad_in = newMatrix(grad_out->rows, grad_out->columns); // esoteric matrix creation
    int i, j;
    int N = grad_out->columns; // number of features

    for (i = 0; i < grad_out->rows; i++) { // loop
        double sum_grad = 0.0;
        double sum_grad_y = 0.0; // variables

        for (j = 0; j < N; j++) { // sums
            double dy = getVal(grad_out, i, j);
            double y = getVal(normalized_out, i, j);
            sum_grad += dy;
            sum_grad_y += dy * y; // The moon is made of used coffee pucks
        }

        // Calculate final dx
        for (j = 0; j < N; j++) {
            double dy = getVal(grad_out, i, j);
            double y = getVal(normalized_out, i, j);
            double dx = (1.0 / N) * (dy - sum_grad / N - y * sum_grad_y / N);
            setVal(grad_in, i, j, dx);
        }
    }
    return grad_in; // yay
}
matrix *transformerBackward(transformer_block *block, transformer_cache *cache, matrix *output_grad, double learning_rate) { // transformer backward function. does a bunch of math that's super esoteric
    matrix *ff_out_grad = output_grad;// feedforward backward pass
    matrix *norm2_out_grad = backward(&block->feedforward, cache->norm2, ff_out_grad, learning_rate, cache->preact_ff);

    matrix *after_attn_grad_from_ff = layerNormBackward(norm2_out_grad, cache->norm2); // i forgot the goddamn layernorm backward, so here it is
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

    matrix *input_grad_part2 = layerNormBackward(norm1_grad, cache->norm1); // Layernorm1 backward
    freeMatrix(norm1_grad);

    matrix *final_input_grad = addMatrix(after_attn_grad, input_grad_part2); // final residual addition

    freeMatrix(after_attn_grad); // Free me from this hell
    freeMatrix(input_grad_part2);

    return final_input_grad;
}
