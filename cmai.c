#include <stdio.h> // From the ashes of fallen kingdoms, the First Dragon forged the Runes of C. We are but mortals wielding their power. Now speak your incantations and see what stirs.
#include <stdlib.h> // same but for stdlib.h
#include <math.h> // BOOOORINGGGGGG

#define LN_EPSILON 1e-5 // super small number. is used later on



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
    if (row < 0 || row >= m->rows || col < 0 || col >= m->columns) return -1; // bounds check
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

double relu(double x) { // makes a function
    if (x < 0) return 0; // If x is less than 0, return 0
    return x; // return x
}

double sigmoid(double x) { // makes the sigmoid function
    return 1.0 / (1.0 + exp(-x)); // return one difided by the sum of one plus the exponent of negative x
}


/* --------------------------------------------- *\
   | Layers and other stuff on top of matrices |
\* --------------------------------------------- */


// < From here, the math gets more complicated, i copypasted some of it, i don't really know what it does but i guess it works >

typedef struct { // New struct!!! yay
    matrix *weights; // weights matrix
    matrix *biases; // biases matrix
    double (*activation)(double); // pointer to an activation function
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

double relu_derivative(double x) { // relu but derivative :thumbsup:
    return x > 0 ? 1.0 : 0.0; // math.
}

double sigmoid_derivative(double x) { // sigmoid but derivative
    double s = sigmoid(x); // sigmoid
    return s * (1.0 - s); // derivative
}

double activation_derivative(double (*func)(double), double x) { // absolutely, yes, of course
    if (func == relu) return relu_derivative(x);
    if (func == sigmoid) return sigmoid_derivative(x);
    return 0;
}


matrix *backward(layer *l, matrix *input, matrix *output_grad, double learning_rate, matrix *preact) { // i copy pasted this. sue me. i kinda understand it but at the same time i dont.
    // output_grad is the gradient from the loss function                                              // backprop is hard.
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
    for (i = 0; i < l->weights->rows; i++) { // updates weights
        for (j = 0; j < l->weights->columns; j++) {
            double w = getVal(l->weights, i, j);
            double grad = getVal(weight_grad, i, j);
            setVal(l->weights, i, j, w - learning_rate * grad); // sets values
        }
    }

    // update biases (sum each column of activation_grad)
    for (j = 0; j < activation_grad->columns; j++) { // updates biases
        double bias_grad = 0;
        for (i = 0; i < activation_grad->rows; i++) {
            bias_grad += getVal(activation_grad, i, j); // magic
        }
        double b = getVal(l->biases, 0, j);
        setVal(l->biases, 0, j, b - learning_rate * bias_grad); // sets values
    }

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
    mean = 0; // sets mean to 0

    for (i = 0; i < m->rows; i++) { // loopy loop loop!!! loops through rows
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

matrix *attention(matrix *queries, matrix *keys, matrix *values) { // super important. takes queries, keys and values
    matrix *keys_t = transposeMatrix(keys); // transpose keys
    matrix *scores = multiplyMatrix(queries, keys_t); // scores = queries * keys^T how much does each query match each key?


    double scale = 1.0 / sqrt(keys->columns); // scale by sqrt(key_dimension), also because everybody does it
    int i; // interestingly unknown declaration, please tell me what this means
    for (i = 0; i < scores->rows * scores->columns; i++) // looplooplooploop
        scores->data[i] *= scale; // multiplies scores data[i] by the

    softmax(scores); // apply softmax to scores

    matrix *output = multiplyMatrix(scores, values); // output = scores * values
    freeMatrix(keys_t); // free the transposed keys

    return output; // returns the returning outputtive output
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
    block->wq.activation = relu; // projections usually use relu so we'll use that

    block->wk.weights = newMatrix(input_size, input_size);
    block->wk.biases = newMatrix(1, input_size);
    block->wk.activation = relu;

    block->wv.weights = newMatrix(input_size, input_size);
    block->wv.biases = newMatrix(1, input_size);
    block->wv.activation = relu;

    block->wo.weights = newMatrix(input_size, input_size);
    block->wo.biases = newMatrix(1, input_size);
    block->wo.activation = relu;

    // Randomize the projection weights
    int i;
    for (i = 0; i < input_size * input_size; i++) {
        block->wq.weights->data[i] = (rand() % 100) / 100.0 - 0.5;
        block->wk.weights->data[i] = (rand() % 100) / 100.0 - 0.5;
        block->wv.weights->data[i] = (rand() % 100) / 100.0 - 0.5;
        block->wo.weights->data[i] = (rand() % 100) / 100.0 - 0.5;
    }

    block->feedforward.weights = newMatrix(input_size, hidden_size); // sets the weights to a matrix with rows(input) and columns(hidden size)
    block->feedforward.biases = newMatrix(1, hidden_size); // sets the biases to a matrix of 1 row and hidden_size columns
    block->feedforward.activation = activation; // sets the activation to activation

    for (i = 0; i < input_size * hidden_size; i++) // for loop
        block->feedforward.weights->data[i] = (rand() % 100) / 100.0 - 0.5; // sets the  weights randomly

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

matrix *transformerForward(matrix *input, transformer_block *block) { // forward pass (which i HATE) on a transformer blocks
    matrix *norm1 = layerNorm(input); // attention OUTput

    matrix *preact_q = newMatrix(norm1->rows, block->wq.weights->columns); // making new matrices for the projections
    matrix *preact_k = newMatrix(norm1->rows, block->wk.weights->columns);
    matrix *preact_v = newMatrix(norm1->rows, block->wv.weights->columns);
    matrix *Q = forward(&block->wq, norm1, preact_q); // ughhh
    matrix *K = forward(&block->wk, norm1, preact_k);
    matrix *V = forward(&block->wv, norm1, preact_v);

    matrix *attn_out = attention(Q, K, V); // after attention

    matrix *preact_o = newMatrix(attn_out->rows, block->wo.weights->columns); // preact output
    matrix *proj_out = forward(&block->wo, attn_out, preact_o); // projection output

    matrix *after_attn = addMatrix(input, proj_out); // after attention
    freeMatrix(norm1); // freeing time
    freeMatrix(preact_q);
    freeMatrix(Q);
    freeMatrix(preact_k);
    freeMatrix(K);
    freeMatrix(preact_v);
    freeMatrix(V);
    freeMatrix(attn_out);
    freeMatrix(preact_o);
    freeMatrix(proj_out);

    matrix *norm2 = layerNorm(after_attn); // preact matrix creation
    matrix *preact = newMatrix(norm2->rows, block->feedforward.biases->columns); // ff_out matrix creation
    matrix *ff_out = forward(&block->feedforward, norm2, preact); // adds after_attn and ff_out together

    matrix *output = addMatrix(after_attn, ff_out);
    freeMatrix(after_attn); // frees everything else
    freeMatrix(norm2);
    freeMatrix(preact);
    freeMatrix(ff_out);

    return output; // returns the output
}

typedef struct { // embedding struct
    matrix *weights; // giant lookup table for the embeddings
    int vocab_size; // vocab size
    int embedding_dim; // embedding dimensions
} embedding; // alias

embedding *createEmbedding(int vocab_size, int embedding_dim) { // new function, makes an embedding, takes vocab size and the dimensions :thumbsup:
    embedding *e = malloc(sizeof(embedding)); // allocates
    if (e == NULL) return NULL; // error handling

    e->vocab_size = vocab_size; // sets vocab size
    e->embedding_dim = embedding_dim; // sets embedding dimensions
    e->weights = newMatrix(vocab_size, embedding_dim); // sets weights

    int i; // i
    for (i = 0; i < vocab_size * embedding_dim; i++) { // for loop
        e->weights->data[i] = (rand() % 100) / 100.0 - 0.5; // randomizes dimensions
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
