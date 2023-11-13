#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <sndfile.h>
#include <string>
const int INPUT_SIZE = 38;
const int HIDDEN_SIZE = 8;
const int NUM_LAYERS = 4;
const int BATCH_SIZE = 1;
const int NUM_SAMPLES = 100;

// Struct representing a 2D array
struct Array2D {
    int x;
    int y;
    double* data;
};

// Struct that stores a 1D vector
struct Vector {
    int size;
    double* data;
};

// Function to initialize a 2D array
struct Array2D* createArray2D(int width, int height) {
    struct Array2D* arr = (struct Array2D*)malloc(sizeof(struct Array2D));
    if (arr == NULL) {
      std::cout << "Allocation failed" << std::endl;
      exit(1); // Allocation failed
    }
    arr->x = width;
    arr->y = height;
    arr->data = (double*)malloc(width * height * sizeof(double));
    if (arr->data == NULL) {
         std::cout << "Allocation failed 2" << std::endl;
        free(arr);
        exit(1); // Allocation failed
    }
    return arr;
}

void fillArray2D(struct Array2D* arr, double value) {
    for (int i = 0; i < arr->x * arr->y; i++) {
        arr->data[i] = value;
    }
}

// Function to initialize a 2D array
struct Vector* createVector(int size) {
    struct Vector* arr = (struct Vector*)malloc(sizeof(struct Vector));
    if (arr == NULL) {
      std::cout << "Allocation failed" << std::endl;
      exit(1); // Allocation failed
    }
    arr->size = size;
    arr->data = (double*)malloc(size * sizeof(double));
    if (arr->data == NULL) {
         std::cout << "Allocation failed 2" << std::endl;
        free(arr);
        exit(1); // Allocation failed
    }
    return arr;
}

void fillVector(struct Vector* arr, double value) {
    for (int i = 0; i < arr->size; i++) {
        arr->data[i] = value;
    }
}

void freeVector(struct Vector* arr) {
    free(arr->data);
    free(arr);
}

// Function to return array dimensions as a string
std::string getArray2DShape(struct Array2D* arr) {
    std::string shape = "(" + std::to_string(arr->x) + "," + std::to_string(arr->y) + ")";
    return shape;
}

// Function to return array contents as a string
std::string getArray2DContents(struct Array2D* arr) {
    std::string contents = "";
    for (int i = 0; i < arr->x; i++) {
        for (int j = 0; j < arr->y; j++) {
            contents += std::to_string(arr->data[i * arr->y + j]) + " ";
        }
        contents += "\n";
    }
    return contents;
}

// Function to get the value at a specific 2D index
double get(struct Array2D* arr, int x, int y) {
    if (x < 0 || x >= arr->x || y < 0 || y >= arr->y) {
        // Index out of bounds. Remove this later to improve speed.
        std::cout << "Out of bounds error in get" << std::endl;
        std::cout << "X = " << x << " vs " << arr->x << ", Y = " << y << " vs " << arr->y << std::endl;
        exit(1);
    }
    int index = x + y * arr->x;
    return arr->data[index];
}

// Function to set the value at a specific 2D index
void set(struct Array2D* arr, int x, int y, double value) {
    if (x < 0 || x >= arr->x || y < 0 || y >= arr->y) {
      // Index out of bounds. Remove this later to improve speed.
        std::cout << "Out of bounds error in get" << std::endl;
        std::cout << "X = " << x << " vs " << arr->x << ", Y = " << y << " vs " << arr->y << std::endl;
        exit(1);
    }
    int index = x + y * arr->x;
    arr->data[index] = value; 
}

// Function to free the memory used by the 2D array
void freeArray2D(struct Array2D* arr) {
    free(arr->data);
    free(arr);
}

// Concatenate two 2D arrays along the specified axis
// Result needs to be x by y1 + y2
void concatenateArrays(Array2D* array1, Array2D* array2, int axis, Array2D* result) {
    if (axis == 1) { // Concatenate along columns
        if (array1->x != array2->x) {
            std::cout << "Incompatible dimensions in concatenateArrays" << std::endl;
            std::cout << "(" << array1->x << "," << array1->y << ")" << " != " << "(" << array2->x << "," << array2->y << ")" << std::endl;
            exit(1);
        }

        if (result->x != array1->x || result->y != array1->y + array2->y) {
            std::cout << "Incompatible dimensions in concatenateArrays result" << std::endl;
            std::cout << result->x << " != " << array1->x << " || " << result->y << " != " << array1->y << " + " << array2->y << std::endl;
            exit(1);
        }

        for (int i = 0; i < array1->x; i++) {
            for (int j = 0; j < array1->y; j++) {
                set(result, i, j, get(array1, i, j));
            }
            for (int j = 0; j < array2->y; j++) {
                set(result, i, j + array1->y, get(array2, i, j));
            }
        }
    } else {
        // Handle other cases (e.g., concatenate along rows)
        std::cout << "Incompatible axis in concatenateArrays" << std::endl;
         exit(1);
    }
}

Vector* concatenateVectors(Vector* A, Vector* B, Vector* C)
{
    if (A->size + B->size != C->size)
    {
        std::cout << "Incompatible dimensions in concatenateVectors" << std::endl;
        exit(1);
    }

    for (int i = 0; i < A->size; i++)
    {
        C->data[i] = A->data[i];
    }
    for (int i = 0; i < B->size; i++)
    {
        C->data[i + A->size] = B->data[i];
    }

    return C;
}

// Matrix multiplication (np.matmul)
void matmul(Array2D* A, Array2D* B, Array2D* result) {
    if (A->y != B->x) {
      std::cout << "Incompatible dimensions in matmul" << std::endl;
      std::cout << A->y << " != " << B->x << std::endl;
      std::cout << getArray2DShape(A) << " != " << getArray2DShape(B) << std::endl;
      exit(1);
    }
    
    if (result->x != A->x || result->y != B->y) {
      std::cout << "Incompatible dimensions in matmul result" << std::endl;
        std::cout << result->x << " != " << A->x << " || " << result->y << " != " << B->y << std::endl;
        std::cout << getArray2DShape(A) << " , " << getArray2DShape(B) << " , " << getArray2DShape(result) << std::endl;
      exit(1);
    }

    for (int i = 0; i < A->x; i++) {
        for (int j = 0; j < B->y; j++) {
            double sum = 0.0;
            for (int k = 0; k < A->y; k++) {
                sum += A->data[i * A->y + k] * B->data[k * B->y + j];
            }
            result->data[i * B->y + j] = sum;
        }
    }
}

// Sigmoid activation function
void sigmoid(Array2D* X, Array2D* result) {
   if (X->x != result->x || X->y != result->y) {
      std::cout << "Incompatible dimensions in sigmoid result" << std::endl;
      exit(1);
   }

   for (int i = 0; i < X->x; i++) {
      for (int j = 0; j < X->y; j++) {
         result->data[i * X->y + j] = 1.0 / (1.0 + exp(-X->data[i * X->y + j]));
      }
   }
}

// Sigmoid activation function, but in-place
void _sigmoid(Array2D* X) {
   for (int i = 0; i < X->x; i++) {
      for (int j = 0; j < X->y; j++) {
         X->data[i * X->y + j] = 1.0 / (1.0 + exp(-X->data[i * X->y + j]));
      }
   }
}

// Tanh activation function
void tanh_activation(Array2D* X, Array2D* result) {
   if (X->x != result->x || X->y != result->y) {
      std::cout << "Incompatible dimensions in tanh_activation result" << std::endl;
      exit(1);
   }

   for (int i = 0; i < X->x; i++) {
      for (int j = 0; j < X->y; j++) {
         result->data[i * X->y + j] = tanh(X->data[i * X->y + j]);
      }
   }
}

// Tanh activation function (in-place)
void _tanh_activation(Array2D* X) {

   for (int i = 0; i < X->x; i++) {
      for (int j = 0; j < X->y; j++) {
         X->data[i * X->y + j] = tanh(X->data[i * X->y + j]);
      }
   }
}

// Softmax activation function
// X is an array of x by y
// result is also x by y
// exp_X is also x by y
// exp_X_sum is x by 1
void softmax(Array2D* X, Array2D* result, Array2D* exp_X, Array2D* exp_X_sum) {
   if (X->x != result->x || X->y != result->y) {
      std::cout << "Incompatible dimensions in softmax result" << std::endl;
      exit(1);
   }
   if (X->x != exp_X->x || X->y != exp_X->y) {
      std::cout << "Incompatible dimensions in softmax exp_X" << std::endl;
      exit(1);
   }
   if (X->x != exp_X_sum->x || exp_X_sum->y != 1) {
      std::cout << "Incompatible dimensions in softmax exp_X_sum" << std::endl;
      exit(1);
   }

   // Calculate exp(X)
    for (int i = 0; i < X->x; i++) {
        for (int j = 0; j < X->y; j++) {
            set(exp_X, i, j, exp(get(X, i, j)));
        }
    }

    // Calculate the sum of exp(X) along axis 1 (rows)
    for (int i = 0; i < X->x; i++) {
        double sum = 0.0;
        for (int j = 0; j < X->y; j++) {
            sum += get(exp_X, i, j);
        }
        set(exp_X_sum, i, 0, sum);
    }

    // Calculate softmax values
    for (int i = 0; i < X->x; i++) {
        for (int j = 0; j < X->y; j++) {
            double numerator = get(exp_X, i, j);
            double denominator = get(exp_X_sum, i, 0);
            set(result, i, j, numerator / denominator);
        }
    }
}

// Softmax activation function
// X is an array of x by y
// result is also x by y
// exp_X is also x by y
// exp_X_sum is x by 1
void _softmax(Array2D* X, Array2D* exp_X, Array2D* exp_X_sum) {
   if (X->x != exp_X->x || X->y != exp_X->y) {
      std::cout << "Incompatible dimensions in _softmax exp_X" << std::endl;
      exit(1);
   }
   if (X->x != exp_X_sum->x || exp_X_sum->y != 1) {
      std::cout << "Incompatible dimensions in _softmax exp_X_sum" << std::endl;
      exit(1);
   }

   // Calculate exp(X)
    for (int i = 0; i < X->x; i++) {
        for (int j = 0; j < X->y; j++) {
            set(exp_X, i, j, exp(get(X, i, j)));
        }
    }

    // Calculate the sum of exp(X) along axis 1 (rows)
    for (int i = 0; i < X->x; i++) {
        double sum = 0.0;
        for (int j = 0; j < X->y; j++) {
            sum += get(exp_X, i, j);
        }
        set(exp_X_sum, i, 0, sum);
    }

    // Calculate softmax values
    for (int i = 0; i < X->x; i++) {
        for (int j = 0; j < X->y; j++) {
            double numerator = get(exp_X, i, j);
            double denominator = get(exp_X_sum, i, 0);
            set(X, i, j, numerator / denominator);
        }
    }
}

// Element wise multiplication
void hadamard_product(Array2D* X, Array2D* Y, Array2D* Z)
{
    if (X->x != Y->x || X->y != Y->y) {
      std::cout << "Incompatible dimensions in hadamard_product input" << std::endl;
      std::cout << getArray2DShape(X) << " != " << getArray2DShape(Y) << std::endl;
      exit(1);
   }
   if (X->x != Z->x || Z->y != X->y) {
      std::cout << "Incompatible dimensions in hadamard_product result" << std::endl;
      exit(1);
   }

   for (int i = 0; i < X->x; i++)
   {
    for (int j = 0; j < X->y; j++)
    {
        set(Z, i, j, get(X, i, j) * get(Y, i, j));
    }
   }
}

// Element-wise addition of two 2D arrays
void add_arrays2D(Array2D* X, Array2D* Y, Array2D* Z) {
   if (X->x != Y->x || X->y != Y->y) {
      std::cout << "Incompatible dimensions in add_arrays2D" << std::endl;
      exit(1);
   }
   if (X->x != Z->x || X->y != Z->y) {
      std::cout << "Incompatible dimensions in add_arrays2D" << std::endl;
      exit(1);
   }

   for (int i = 0; i < X->x * X->y; i++) {
         Z->data[i] = X->data[i] + Y->data[i];
   }
}

void add_arrays2D(Array2D* X, Array2D* Y, Array2D* Z, Array2D* output) {
   if (X->x != Y->x || X->y != Y->y) {
      std::cout << "Incompatible dimensions in add_arrays2D" << std::endl;
      exit(1);
   }
   if (X->x != Z->x || X->y != Z->y) {
      std::cout << "Incompatible dimensions in add_arrays2D" << std::endl;
      exit(1);
   }
   if (X->x != output->x || X->y != output->y) {
      std::cout << "Incompatible dimensions in add_arrays2D output" << std::endl;
      exit(1);
   }

   for (int i = 0; i < X->x * X->y; i++) {
        output->data[i] = X->data[i] + Y->data[i] + Z->data[i];
   }
}

// X and Y are hidden_size by batch_size
// Z is hidden_size long.
void add_arrays2D(Array2D* X, Vector* bias, Array2D* Y, Vector* bias2, Array2D* output) {
   if (X->x != Y->x || X->y != Y->y) {
      std::cout << "Incompatible dimensions in add_arrays2D" << std::endl;
      std::cout << getArray2DShape(X) << " != " << getArray2DShape(Y) << std::endl;
      exit(1);
   }
   if (X->x != bias2->size || Y->x != bias2->size) {
      std::cout << "Incompatible dimensions in add_arrays2D" << std::endl;
      exit(1);
   }
   if (X->x != output->x || X->y != output->y) {
      std::cout << "Incompatible dimensions in add_arrays2D output" << std::endl;
      exit(1);
   }

   for (int i = 0; i < X->x; i++) {
        for (int j = 0; j < X->y; j++) {
            output->data[i * X->y + j] = X->data[i * X->y + j] + bias->data[i] + Y->data[i * X->y + j] + bias2->data[i];
        }
   }
}

// Extract a horizontal slice of a 2D array (a single row)
Vector* verticalSlice(Array2D* array, int row, Vector* slice) {
    if (row < 0 || row >= array->x) {
        std::cout << "Incompatible index in verticalSlice" << std::endl;
      exit(1);
    }
    if (slice->size != array->y) {
        std::cout << "Incompatible dimensions in verticalSlice" << std::endl;
      exit(1);
    }

    for (int i = 0; i < array->y; i++) {
        double value = get(array, row, i);
        slice->data[i] = value;
    }

    //std::cout << "Returning slice" << std::endl;

    return slice;
}

// Extract a horizontal slice of a 2D array
Array2D* verticalSlice(Array2D* array, int startRow, int size, Array2D* slice) {
    if (startRow < 0 || startRow + size > array->x) {
        std::cout << "Incompatible index in verticalSlice" << std::endl;
      exit(1);
    }
    if (slice->y != array->y || slice->x != size) {
        std::cout << "Incompatible dimensions in verticalSlice" << std::endl;
      exit(1);
    }
    for (int j = 0; j < array->y; j++) {
        for (int i = 0; i < size; i++) {
            double value = get(array, startRow + i, j);
            set(slice, i, j, value);
        }
    }

    //std::cout << "Returning slice" << std::endl;

    return slice;
}

// In PyTorch, weight_ih corresponds to four different weights.
struct LSTM_weights
{
    // Pytorch versions
    Array2D** weight_ih; // Should be NUM_LAYERS by 4 * hidden_size by input_size
    Array2D** weight_hh; // Should be NUM_LAYERS by 4 * hidden_size by hidden_size

    Vector** ibias_input; // Should be NUM_LAYERS by hidden_size. This is b_ii
    Vector** ibias_forget; // Should be NUM_LAYERS by hidden_size. This is b_if
    Vector** ibias_gate; // Should be NUM_LAYERS by hidden_size. This is b_ig
    Vector** ibias_output; // Should be NUM_LAYERS by hidden_size. This is b_io

    Vector** hbias_input; // Should be NUM_LAYERS by hidden_size. This is b_hi
    Vector** hbias_forget; // Should be NUM_LAYERS by hidden_size. This is b_hf
    Vector** hbias_gate; // Should be NUM_LAYERS by hidden_size. This is b_hg
    Vector** hbias_output; // Should be NUM_LAYERS by hidden_size. This is b_ho

};

// Mallocs all the arrays for the weights and biases
LSTM_weights* init_weights(double init)
{
    LSTM_weights* weights = (LSTM_weights*) malloc(sizeof(struct LSTM_weights));

    // Allocate a load of arrays of pointers to 2D arrays

    weights->weight_ih = (Array2D**) malloc(NUM_LAYERS * sizeof(struct Array2D*));
    weights->weight_hh = (Array2D**) malloc(NUM_LAYERS * sizeof(struct Array2D*));

    weights->ibias_input = (Vector**) malloc(NUM_LAYERS * sizeof(struct Vector*));
    weights->ibias_forget = (Vector**) malloc(NUM_LAYERS * sizeof(struct Vector*));
    weights->ibias_gate = (Vector**) malloc(NUM_LAYERS * sizeof(struct Vector*));
    weights->ibias_output = (Vector**) malloc(NUM_LAYERS * sizeof(struct Vector*));

    weights->hbias_input = (Vector**) malloc(NUM_LAYERS * sizeof(struct Vector*));
    weights->hbias_forget = (Vector**) malloc(NUM_LAYERS * sizeof(struct Vector*));
    weights->hbias_gate = (Vector**) malloc(NUM_LAYERS * sizeof(struct Vector*));
    weights->hbias_output = (Vector**) malloc(NUM_LAYERS * sizeof(struct Vector*));

    for (int i = 0; i < NUM_LAYERS; i++)
    {
        if (i == 0)
        {
            weights->weight_ih[i] = createArray2D( 4 * HIDDEN_SIZE, INPUT_SIZE);
            
        }
        else
        {
            weights->weight_ih[i] = createArray2D(4 * HIDDEN_SIZE,HIDDEN_SIZE);
        }
        fillArray2D(weights->weight_ih[i], init);

        weights->weight_hh[i] = createArray2D(4 * HIDDEN_SIZE, HIDDEN_SIZE);
        fillArray2D(weights->weight_hh[i], init);

        weights->ibias_input[i] = createVector(HIDDEN_SIZE);
        fillVector(weights->ibias_input[i], init);

        weights->ibias_forget[i] = createVector(HIDDEN_SIZE);
        fillVector(weights->ibias_forget[i], init);

        weights->ibias_gate[i] = createVector(HIDDEN_SIZE);
        fillVector(weights->ibias_gate[i], init);

        weights->ibias_output[i] = createVector(HIDDEN_SIZE);
        fillVector(weights->ibias_output[i], init);


        weights->hbias_input[i] = createVector(HIDDEN_SIZE);
        fillVector(weights->hbias_input[i], init);

        weights->hbias_forget[i] = createVector(HIDDEN_SIZE);
        fillVector(weights->hbias_forget[i], init);

        weights->hbias_gate[i] = createVector(HIDDEN_SIZE);
        fillVector(weights->hbias_gate[i], init);

        weights->hbias_output[i] = createVector(HIDDEN_SIZE);
        fillVector(weights->hbias_output[i], init);

        
    }
    return weights;
}

void free_LSTM_weights(LSTM_weights* weights)
{
    for (int i = 0; i < NUM_LAYERS; i++)
    {
        freeArray2D(weights->weight_ih[i]);
        freeArray2D(weights->weight_hh[i]);

        freeVector(weights->ibias_input[i]);
        freeVector(weights->ibias_forget[i]);
        freeVector(weights->ibias_gate[i]);
        freeVector(weights->ibias_output[i]);

        freeVector(weights->hbias_input[i]);
        freeVector(weights->hbias_forget[i]);
        freeVector(weights->hbias_gate[i]);
        freeVector(weights->hbias_output[i]);
    }

    free(weights->weight_ih);
    free(weights->weight_hh);

    free(weights->ibias_input);
    free(weights->ibias_forget);
    free(weights->ibias_gate);
    free(weights->ibias_output);

    free(weights->hbias_input);
    free(weights->hbias_forget);
    free(weights->hbias_gate);
    free(weights->hbias_output);

    free(weights);
}

void init_states(Array2D* hidden_state, Array2D* cell_state) { 
    hidden_state = createArray2D(NUM_LAYERS, HIDDEN_SIZE);
    cell_state = createArray2D(NUM_LAYERS, HIDDEN_SIZE);

    for (int i = 0; i < NUM_LAYERS; i++)
    {
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            set(hidden_state, i, j, 0.0);
            set(cell_state, i, j, 0.0);
        }
    }
}

struct LSTM_Working_Memory
{
    Array2D* input_gate;
    Array2D* forget_gate;
    Array2D* output_gate;
    Array2D* gate_gate;

    Array2D* w_times_input;
    Array2D* inp_slice;
    Array2D* forget_slice;
    Array2D* gate_slice;
    Array2D* output_slice;

    Array2D* h_times_state;
    Array2D* h_inp_slice;
    Array2D* h_forget_slice;
    Array2D* h_gate_slice;
    Array2D* h_output_slice;

    Array2D* forget_times_cell;
    Array2D* input_times_gate;
    Array2D* tanh_cell;
};

LSTM_Working_Memory* init_LSTM_Working_Memory(double init)
{
    LSTM_Working_Memory* wm = (LSTM_Working_Memory*) malloc(sizeof(struct LSTM_Working_Memory));
    wm->input_gate = createArray2D(HIDDEN_SIZE, BATCH_SIZE);
    wm->forget_gate = createArray2D(HIDDEN_SIZE, BATCH_SIZE);
    wm->output_gate = createArray2D(HIDDEN_SIZE, BATCH_SIZE);
    wm->gate_gate = createArray2D(HIDDEN_SIZE, BATCH_SIZE);

    wm->w_times_input = createArray2D(HIDDEN_SIZE * 4, BATCH_SIZE);
    wm->inp_slice = createArray2D(HIDDEN_SIZE, BATCH_SIZE);
    wm->forget_slice = createArray2D(HIDDEN_SIZE, BATCH_SIZE);
    wm->output_slice = createArray2D(HIDDEN_SIZE, BATCH_SIZE);
    wm->gate_slice = createArray2D(HIDDEN_SIZE, BATCH_SIZE);

    wm->h_times_state = createArray2D(HIDDEN_SIZE * 4, BATCH_SIZE);
    wm->h_inp_slice = createArray2D(HIDDEN_SIZE, BATCH_SIZE);
    wm->h_forget_slice = createArray2D(HIDDEN_SIZE, BATCH_SIZE);
    wm->h_output_slice = createArray2D(HIDDEN_SIZE, BATCH_SIZE);
    wm->h_gate_slice = createArray2D(HIDDEN_SIZE, BATCH_SIZE);

    wm->forget_times_cell = createArray2D(HIDDEN_SIZE, BATCH_SIZE);
    wm->input_times_gate = createArray2D(HIDDEN_SIZE, BATCH_SIZE);
    wm->tanh_cell = createArray2D(HIDDEN_SIZE, BATCH_SIZE);

    // Initialize all the arrays to 0
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        for (int j = 0; j < BATCH_SIZE; j++)
        {
            set(wm->input_gate, i, j, init);
            set(wm->forget_gate, i, j, init);
            set(wm->output_gate, i, j, init);
            set(wm->gate_gate, i, j, init);
            
            set(wm->inp_slice, i, j, init);
            set(wm->forget_slice, i, j, init);
            set(wm->output_slice, i, j, init);
            set(wm->gate_slice, i, j, init);
            
            set(wm->h_inp_slice, i, j, init);
            set(wm->h_forget_slice, i, j, init);
            set(wm->h_output_slice, i, j, init);
            set(wm->h_gate_slice, i, j, init);

            set(wm->forget_times_cell, i, j, init);
            set(wm->input_times_gate, i, j, init);
            set(wm->tanh_cell, i, j, init);
        }
    }

    for (int i = 0; i < HIDDEN_SIZE * 4; i++)
    {
        for (int j = 0; j < BATCH_SIZE; j++)
        {
            set(wm->w_times_input, i, j, init);
            set(wm->h_times_state, i, j, init);
        }
    }

    return wm;
}

void free_LSTM_Working_Memory(LSTM_Working_Memory* wm)
{
    freeArray2D(wm->input_gate);
    freeArray2D(wm->forget_gate);
    freeArray2D(wm->output_gate);
    freeArray2D(wm->gate_gate);

    freeArray2D(wm->w_times_input);
    freeArray2D(wm->inp_slice);
    freeArray2D(wm->forget_slice);
    freeArray2D(wm->output_slice);
    freeArray2D(wm->gate_slice);

    freeArray2D(wm->h_times_state);
    freeArray2D(wm->h_inp_slice);
    freeArray2D(wm->h_forget_slice);
    freeArray2D(wm->h_output_slice);
    freeArray2D(wm->h_gate_slice);

    freeArray2D(wm->forget_times_cell);
    freeArray2D(wm->input_times_gate);
    freeArray2D(wm->tanh_cell);

    free(wm);
}

Array2D** forward(Array2D** input_sequence, Array2D** hidden_states, Array2D* cell_state, Array2D* prev_cell_state, LSTM_weights* weights, LSTM_Working_Memory* wm)
{
    for (int sample = 0; sample < NUM_SAMPLES; sample++)
    {
        std::cout << std::endl << "Sample: " << sample << std::endl << std::endl;
        for (int layer = 0; layer < NUM_LAYERS; layer++)
        {
            // Get input times input weights
            std::cout << std::endl << "Layer: " << layer << std::endl << std::endl;

            if (sample > 0)
                std::cout << "Hidden state at start: " << std::endl << getArray2DContents(hidden_states[sample-1]) << std::endl;
            std::cout << "Cell state at start: " << std::endl << getArray2DContents(prev_cell_state) << std::endl;

            //std::cout << getArray2DShape(weights->weight_ih[layer]) << std::endl;
            //std::cout << getArray2DShape(input_sequence) << std::endl;


            // Do input times input weights and times previous hidden state
            if (layer == 0)
            {
                // If we're on layer 0, use the input sequence
                matmul(weights->weight_ih[layer],input_sequence[sample], wm->w_times_input);
            }
            else
            {
                matmul(weights->weight_ih[layer],hidden_states[sample], wm->w_times_input);
                
            }

            if (sample == 0)
            {
                // If we're on sample 0, h_(t-1) is all zeros, so just fill the result with zeros
                fillArray2D(wm->h_times_state, 0.0);
            }
            else
            {
                matmul(weights->weight_hh[layer],hidden_states[sample-1], wm->h_times_state);
            }
                
            
            std::cout << "W times input: " << getArray2DContents(wm->w_times_input) << std::endl;
            std::cout << "H times state: " << getArray2DContents(wm->h_times_state) << std::endl;

            // Get the slices of the input times input weights
            verticalSlice(wm->w_times_input, 0, HIDDEN_SIZE, wm->inp_slice);
            verticalSlice(wm->w_times_input, HIDDEN_SIZE, HIDDEN_SIZE, wm->forget_slice);
            verticalSlice(wm->w_times_input, HIDDEN_SIZE * 2, HIDDEN_SIZE, wm->gate_slice);
            verticalSlice(wm->w_times_input, HIDDEN_SIZE * 3, HIDDEN_SIZE, wm->output_slice);

            //std::cout << "Input slice: " << getArray2DContents(wm->inp_slice) << std::endl;

            //std::cout << "Hidden weights: " << std::endl << getArray2DContents(weights->weight_hh[layer]) << std::endl;
            //std::cout << "Hidden Sequence: " << std::endl << getArray2DContents(hidden_state) << std::endl;

            verticalSlice(wm->h_times_state, 0, HIDDEN_SIZE, wm->h_inp_slice);
            verticalSlice(wm->h_times_state, HIDDEN_SIZE, HIDDEN_SIZE, wm->h_forget_slice);
            verticalSlice(wm->h_times_state, HIDDEN_SIZE * 2, HIDDEN_SIZE, wm->h_gate_slice);
            verticalSlice(wm->h_times_state, HIDDEN_SIZE * 3, HIDDEN_SIZE, wm->h_output_slice);

            // Input gate
            add_arrays2D(wm->inp_slice, weights->ibias_input[layer], wm->h_inp_slice, weights->hbias_input[layer], wm->input_gate);
            std::cout << "Protosigmoid input: " << std::endl << getArray2DContents(wm->input_gate) << std::endl;
            _sigmoid(wm->input_gate);
            std::cout << "Input gate: " << std::endl << getArray2DContents(wm->input_gate) << std::endl;

            

            // Forget gate
            add_arrays2D(wm->forget_slice, weights->ibias_forget[layer], wm->h_forget_slice, weights->hbias_forget[layer], wm->forget_gate);
            std::cout << "Protosigmoid forget: " << std::endl << getArray2DContents(wm->forget_gate) << std::endl;
            _sigmoid(wm->forget_gate);
            std::cout << "Forget gate: " << std::endl << getArray2DContents(wm->forget_gate) << std::endl;

            // Gate gate
            add_arrays2D(wm->gate_slice, weights->ibias_gate[layer], wm->h_gate_slice, weights->hbias_gate[layer], wm->gate_gate);
            std::cout << "Protosigmoid gate: " << std::endl << getArray2DContents(wm->gate_slice) << std::endl;
            _tanh_activation(wm->gate_gate);
            std::cout << "Gate gate: " << std::endl << getArray2DContents(wm->gate_gate) << std::endl;

            // output gate

            add_arrays2D(wm->output_slice, weights->ibias_output[layer], wm->h_output_slice, weights->hbias_output[layer], wm->output_gate);
            std::cout << "Protosigmoid output: " << std::endl << getArray2DContents(wm->output_gate) << std::endl;
            _sigmoid(wm->output_gate);
            std::cout << "Output gate: " << std::endl << getArray2DContents(wm->output_gate) << std::endl;
            
            // Calc new cell state
            hadamard_product(wm->forget_gate, prev_cell_state, wm->forget_times_cell);
            hadamard_product(wm->input_gate, wm->gate_gate, wm->input_times_gate);
            add_arrays2D(wm->forget_times_cell, wm->input_times_gate, cell_state);

            // Calc new hidden state
            tanh_activation(cell_state, wm->tanh_cell);
            hadamard_product(wm->output_gate, wm->tanh_cell, hidden_states[sample]);

            std::cout << "Hidden state: " << std::endl << getArray2DContents(hidden_states[sample]) << std::endl;
            std::cout << "Cell state: " << std::endl << getArray2DContents(cell_state) << std::endl;
        }
        // Move current cell state to previous cell state
        Array2D* temp = prev_cell_state; // Just a pointer. No malloc or computation happening here.
        prev_cell_state = cell_state;
        cell_state = temp;
    }
    
    return hidden_states;
}

int main()
{
    Array2D** input_sequence = (Array2D**) malloc(NUM_SAMPLES * sizeof(struct Array2D*));

    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        input_sequence[i] = createArray2D(INPUT_SIZE, BATCH_SIZE);
        
        fillArray2D(input_sequence[i], .1);
    }

    Array2D* cell_state = createArray2D(HIDDEN_SIZE, BATCH_SIZE);
    Array2D* prev_cell_state = createArray2D(HIDDEN_SIZE, BATCH_SIZE);

    Array2D** hidden_states = (Array2D**) malloc(NUM_SAMPLES * sizeof(struct Array2D*));

    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        hidden_states[i] = createArray2D(HIDDEN_SIZE, BATCH_SIZE);
        
        fillArray2D(hidden_states[i], 0);
    }


    LSTM_weights* weights = init_weights(.1);

    LSTM_Working_Memory* wm = init_LSTM_Working_Memory(0);
    Array2D** out = forward(input_sequence, hidden_states, cell_state, prev_cell_state, weights, wm);

    std::cout << "Output: " << std::endl << std::endl;

    free_LSTM_weights(weights);
    free_LSTM_Working_Memory(wm);

    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        std::cout << "Sample " << i  << ": " << std::endl << getArray2DContents(out[i]) << std::endl;
        freeArray2D(input_sequence[i]);
        freeArray2D(hidden_states[i]);
    }
    
    freeArray2D(cell_state);
    freeArray2D(prev_cell_state);
    
}