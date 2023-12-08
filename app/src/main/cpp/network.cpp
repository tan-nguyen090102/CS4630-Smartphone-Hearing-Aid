#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "network.h"
#include <vector>

#include <chrono>
#include <ctime>

#include <cstring>
#include <sstream>
#include <jni.h>

void print_array(double* arr, int start, int end, std::string name) {
  std::cout << name << std::endl;
  for (int i=start; i < end; i++) {
    printf("%i:%f\n", i, arr[i]);
  }
}

// Loads a .wav file into an array of doubles.
/*SF_INFO loadWavFile(char* filename, std::vector<double> &audioData) {
    // Open the .wav file
    SF_INFO sfinfo;
    SNDFILE* sndfile = sf_open(filename, SFM_READ, &sfinfo);
    if (!sndfile) {
        std::cerr << "Error opening input file: " << sf_strerror(sndfile) << std::endl;
        exit(1);
    }

    // Read the audio data
    const sf_count_t numFrames = sfinfo.frames;
    audioData = std::vector<double>(numFrames);

    sf_count_t numFramesRead = sf_read_double(sndfile, audioData.data(), numFrames);
    if (numFramesRead != numFrames) {
        std::cerr << "Error reading audio data: " << sf_strerror(sndfile) << std::endl;
        sf_close(sndfile);
        exit(1);
    }

    // Close the file and return the audio data
    sf_close(sndfile);
    return sfinfo;
}*/

// Converts a vector of doubles to an array of doubles.
void vectorToArray(std::vector<double> input, double* output, int max_length)
{
  if (max_length == -1)
  {
    max_length = input.size();
  }

  int min_length = std::min(int(input.size()), max_length);
  for (int i = 0; i < min_length; i++)
  {
    output[i] = input[i];
  }
}


void conv1d_unoptimized(double* input, double* weight, double bias, double* output,
            int input_size, int output_size, int kernel_size, int stride) 
{
    for (int i = 0; i < output_size; i++) 
    {
        output[i] += bias; // Add bias

        for (int j = 0; j < kernel_size; j++) 
        {
            if (weight[j] != 0)
            {
              int input_idx = i * stride + j;
              if (input_idx >= 0 && input_idx < input_size) 
              {
                  output[i] += input[input_idx] * weight[j];
              }
            }
        }
    }
}

void conv1d_loops_flipped(double* input, double* weight, double* output,
            int input_size, int output_size, int kernel_size, int stride) 
{
    int i, j, input_idx;

    for (j = 0; j < kernel_size; j++) 
    {
        if (weight[j] != 0)
        {
            for (i = 0; i < output_size; i++) 
            {
                input_idx = i * stride + j;
                output[i] += input[input_idx] * weight[j];
            }
        }
    }
}

void conv1d_shifted(double* input, double* weight, double bias, double* output,
            int input_size, int output_size, int kernel_size, int stride) 
{
    for (int i = 0; i < output_size; i++) 
    {
        output[i] += bias; // Add bias
        for (int j = 0; j < kernel_size; j++) 
        {
            if (weight[j] != 0)
            {
              int input_idx = (i+1) * stride + j;
              if (input_idx >= 0 && input_idx < input_size) 
              {
                  output[i] += input[input_idx] * weight[j];
              }
            }
        }
    }
}

double ReLU(double x) {
    return x > 0 ? x : 0;
}

void ReLU(double* x, int input_length, double* output) 
{
  for (int i=0; i < input_length; i++)
  {
    output[i] = x[i] > 0 ? x[i] : 0;
  }
}

// In place version of ReLU
void ReLU_(double* x, int input_length) 
{
  for (int i=0; i < input_length; i++)
  {
    x[i] = x[i] > 0 ? x[i] : 0;
  }
}

double Sigmoid(double x) {
    return (1 / (1 + exp(-x)));
}

void Sigmoid(double* x, int inp_size, double* output) 
{
    int i;
    for (i=0; i < inp_size; i++) 
    {
        output[i] = (1 / (1 + exp(-x[i])));
    }
}

// GLU activation function
// Returns an array of size inp_size / 2
void GLU(double* inp, int inp_size, double* output) {
    int i;
    int max = inp_size / 2;
    for (i=0; i < max; i++) {
        output[i] = inp[i] * Sigmoid(inp[i + max]);
    }
}

// GLU activation function
// Takes an N * K array, returns an N * K/2 array.
// K should be even
void GLU_split(double* inp, int N, int K, double* output) {
    int i;
    int j;
    int max = K / 2;

    //printf("GLU_split: N: %i, K: %i\n, max: %i", N, K, max);
    for (i=0; i < max; i += 1) 
    {
        for (j=0; j < N; j++)
        {
            output[i * N + j] = inp[i * N + j] * Sigmoid(inp[(max + i) * N + j]);
            //printf("GLU filling %i by matching %i with %i. Result:%f\n", i * N + j, i * N + j, (max + i) * N + j, output[i * N + j]);
        }
        //output[i] = inp[i] * Sigmoid(inp[i + max]);S
    }
    //printf("GLU_split: N: %i, K: %i\n, max: %i", N, K, max);
}

// Assumes that the padding of zeros is ZEROS, and the input size is VALID_LENGTH.
// Applies upsampling twice.
void double_upsample2_valid(double* inp, double* output, WorkingMemory* wm)
{
    // Before convolution, we need to include 112 / 2 = 56 zeros on each side of the input.
    // Use the padded_input array in the working memory struct. This should already have 56 zeros on each side.
    int i;
    for (i=ZEROS; i < VALID_LENGTH + ZEROS; i++)
    {
      wm->padded_upsample_input[i] = inp[i - ZEROS];
    }
    //printf("1\n");


    conv1d_shifted(wm->padded_upsample_input, kernel_upsample, 0, wm->upsample_working, VALID_LENGTH + 2*ZEROS, VALID_LENGTH, 112, 1);

    int output_size = (VALID_LENGTH) * 2;

    //printf("2\n");
    
    // Interweave
    for (int i=0; i < output_size; i++) 
    {
        if (i % 2 == 0) 
        {
            output[i] = inp[(i) / 2];
        }
        else 
        {
            output[i] = wm->upsample_working[(i) / 2];
        }
    }

    //printf("3\n");

    for (i = 0; i < ZEROS; i++)
    {
        wm->padded_upsample_double[i] = 0;
    }
    // Second upsample
    for (i=ZEROS; i < 2*VALID_LENGTH + ZEROS; i++) 
        wm->padded_upsample_double[i] = output[i - ZEROS];
    for (i = 2*VALID_LENGTH + ZEROS; i < 2*VALID_LENGTH + 2*ZEROS; i++)
    {
        wm->padded_upsample_double[i] = 0;
    }
    //printf("4\n");
    conv1d_shifted(wm->padded_upsample_double, kernel_upsample, 0, wm->upsample_working_double, 2 * (VALID_LENGTH + ZEROS), VALID_LENGTH * 2, 112, 1);

    int double_output_size = (VALID_LENGTH) * 4;
    
    //printf("5\n");
    // Interweave
    for (int i=0; i < double_output_size; i++) 
    {
        if (i % 2 == 0) 
        {
            output[i] = wm->padded_upsample_double[i / 2 + ZEROS];
        }
        else 
        {
            output[i] = wm->upsample_working_double[(i) / 2];
        }
    }

}

// Outputs a 2x downsampled version of the input.
// Length of output is inp_length / 2.
// Assumes that the padding of zeros is ZEROS
void downsample2_consts(double* inp, double* output, WorkingMemory* wm, int current_size)
{   
    // Divide into evens and odds.
    // Half input 1 is the evens, 2 is the odds.

    int i;
    for (i = 0; i < current_size; i++) 
    {
        if (i % 2 == 0)
        { // Evens
            wm->half_input_one[i / 2] = inp[i];
        }
        else 
        { // Odds
            wm->half_input_two[(i - 1) / 2] = inp[i];
        }
    }

    if (INPUT_SIZE % 2 == 1)
      wm->half_input_one[current_size/2 - 1] = 0;
    
    //print_array(wm.half_input_one, 0, current_size/2, "evens");
    //print_array(wm.half_input_two, 0, current_size/2, "odds");

    for (i = 0; i < ZEROS ; i++)
    {
      wm->padded_half_input[i] = 0;
      wm->padded_half_input[i + current_size/2 +  ZEROS] = 0;
    }
      

    // Pad the input to the conv1d function. 
    for (i=ZEROS; i < current_size/2 + ZEROS; i++) 
      wm->padded_half_input[i] = wm->half_input_two[i - ZEROS];
    
    for (i = 0; i < current_size/2 ; i++)
      wm->half_input_two[i] = 0;

    // Convolve the padded input with the kernel.
    // Don't need the odd input anymore as that's in wm.padded_half_input, so we use it as the output.
    //conv1d_loops_flipped(wm.padded_half_input, kernel_downsample, 0, wm.half_input_two, (current_size/2) + 2 * ZEROS, current_size / 2, 112, 1);
    conv1d_loops_flipped(wm->padded_half_input, kernel_downsample, wm->half_input_two, (current_size/2) + 2 * ZEROS, current_size / 2, 112, 1);
    
    //print_array(wm.half_input_two, 0, current_size/2, "Conv output");
    // Now add the even and conv_outputs together, then halve them.
    for (i=0; i < current_size / 2; i++) 
    {
        output[i] = (wm->half_input_one[i] + wm->half_input_two[i]) * 0.5;
    }
    //print_array(output, 0, current_size/2, "half_input_two");
    //exit(1);
}


// Calculate multi-channel to multi-channel 1d convolution
// input has size [input_channels, input_size]
// Weight has size [output_channels, input_channels, kernel_size]
void conv1dChannels(double* input, double* weight, double* bias, double* output,
            int input_size, int output_size, int kernel_size, int input_channels, int output_channels, int stride)
{
  int in, out, i;
  // Initialize output channel to zeros
    for (i = 0; i < output_size * output_channels; i++)
    {
      output[i] = 0.0;
    }
    
    for (out = 0; out < output_channels; out++)
    {
        double* channel_out = &output[out * output_size];
        
        //std::cout << "out: " << out << std::endl;
        for (in = 0; in < input_channels; in++)
        {
            //std::cout << "in: " << out << std::endl;
            double* current_inp_channel = &input[in * input_size];
            double* current_weight = &weight[out * input_channels * kernel_size + in * kernel_size];
            
            //conv1d_unoptimized(current_inp_channel, current_weight, 0, channel_out, input_size, output_size, kernel_size, stride);
            //std::cout << "Convolving inp element " << (in * input_size) << " to " << (in + 1) * input_size - 1 << " with weight " << out * input_channels * kernel_size + in * kernel_size << std::endl;
            conv1d_loops_flipped(current_inp_channel, current_weight, channel_out, input_size, output_size, kernel_size, stride);
        }
        if (bias != NULL)
        {
          for (i = 0; i < output_size; i++) 
          {
            //std::cout << "Applying bias number " << out << " to " << i + (out * output_size) << std::endl;
            channel_out[i] += bias[out];
          }
        }
    }
}

// Calculate 1d transpose convolution
void conv1dTranspose(double* input, double* kernel, double* output,
                      int input_size, int kernel_size, int output_size,
                      int stride) {
    int i, k;
    
    // Perform Conv1dTranspose
    for (i = 0; i < input_size; i++)
    { 
      int i_prime = i * stride;
      for (k = 0; k < kernel_size; k++)
      {
        output[i_prime + k] += input[i] * kernel[k];
      }
    }
}

// Calculate multi-channel to multi-channel 1d convolution transpose
// input has size [input_channels, input_size]
// Weight has size [output_channels, input_channels, kernel_size]
void conv1dTransposeChannels(double* input, double* weight, double* bias, double* output,
            int input_size, int kernel_size, int input_channels, int output_channels, int stride)
{
    int i, out, in;
    int output_size = (input_size - 1) * stride + kernel_size;
    if (bias != NULL)
    {
        for (int out = 0; out < output_channels; out++)
        {
            for (int i = 0; i < output_size; i++)
            {
                output[out * output_size + i] = bias[out];
            }
        }
    }
    else
    {
      for (int i = 0; i < output_size * output_channels; i++)
      {
          output[i] = 0.0;
      }
    }

    for (int out = 0; out < output_channels; out++)
    {
        for (int in = 0; in < input_channels; in++)
        {
            double* current_inp_channel = &input[in * input_size];
            double* current_weight = &weight[in * output_channels * kernel_size + out * kernel_size];
            double* channel_out = &output[out * output_size];

            conv1dTranspose(current_inp_channel, current_weight, channel_out, input_size, kernel_size, output_size, stride);
        }
    }
}

Array2D** LSTM_forward(LSTM_weights* weights, LSTM_Working_Memory* wm)
{
    for (int sample = 0; sample < LSTM_NUM_SAMPLES; sample++)
    {
        for (int layer = 0; layer < LSTM_NUM_LAYERS; layer++)
        {

          if (layer == 0)
          {
              // If we're on layer 0, use the input sequence
              matmul(weights->weight_ih[layer],wm->input_sequence[sample], wm->w_times_input);
              //std::cout << "w times input: " << std::endl << getArray2DContents(wm->w_times_input) << std::endl;
          }
          else
          {
            // Otherwise, we use the hidden state of the corresponding layer on the previous sample
              matmul(weights->weight_ih[layer],wm->hidden_states[layer-1], wm->w_times_input);
          }

          if (sample == 0)
          {
            // If we're on sample 0, h_(t-1) is all zeros, so just fill the result with zeros
            fillArray2D(wm->h_times_state, 0.0);
          }
          else
          {
            matmul(weights->weight_hh[layer], wm->hidden_states[layer], wm->h_times_state);
          }
            // Do input times input weights and times previous hidden state
            
                
            
            //std::cout << "W times input: " << getArray2DContents(wm->w_times_input) << std::endl;
            //std::cout << "H times state: " << getArray2DContents(wm->h_times_state) << std::endl;

            // Get the slices of the input times input weights
            verticalSlice(wm->w_times_input, 0, LSTM_HIDDEN_SIZE, wm->inp_slice);
            verticalSlice(wm->w_times_input, LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE, wm->forget_slice);
            verticalSlice(wm->w_times_input, LSTM_HIDDEN_SIZE * 2, LSTM_HIDDEN_SIZE, wm->gate_slice);
            verticalSlice(wm->w_times_input, LSTM_HIDDEN_SIZE * 3, LSTM_HIDDEN_SIZE, wm->output_slice);

            //std::cout << "Input slice: " << getArray2DContents(wm->inp_slice) << std::endl;

            //std::cout << "Hidden weights: " << std::endl << getArray2DContents(weights->weight_hh[layer]) << std::endl;
            //std::cout << "Hidden Sequence: " << std::endl << getArray2DContents(hidden_state) << std::endl;

            verticalSlice(wm->h_times_state, 0, LSTM_HIDDEN_SIZE, wm->h_inp_slice);
            verticalSlice(wm->h_times_state, LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE, wm->h_forget_slice);
            verticalSlice(wm->h_times_state, LSTM_HIDDEN_SIZE * 2, LSTM_HIDDEN_SIZE, wm->h_gate_slice);
            verticalSlice(wm->h_times_state, LSTM_HIDDEN_SIZE * 3, LSTM_HIDDEN_SIZE, wm->h_output_slice);

            //std::cout << "HInput slice: " << getArray2DContents(wm->h_inp_slice) << std::endl;

            // Input gate
            add_arrays2D(wm->inp_slice, weights->ibias_input[layer], wm->h_inp_slice, weights->hbias_input[layer], wm->input_gate);
            //std::cout << "Protosigmoid input: " << std::endl << getArray2DContents(wm->input_gate) << std::endl;
            _sigmoid(wm->input_gate);
            //std::cout << "Input gate: " << std::endl << getArray2DContents(wm->input_gate) << std::endl;

            

            // Forget gate
            add_arrays2D(wm->forget_slice, weights->ibias_forget[layer], wm->h_forget_slice, weights->hbias_forget[layer], wm->forget_gate);
            //std::cout << "Protosigmoid forget: " << std::endl << getArray2DContents(wm->forget_gate) << std::endl;
            _sigmoid(wm->forget_gate);
            //std::cout << "Forget gate: " << std::endl << getArray2DContents(wm->forget_gate) << std::endl;

            // Gate gate
            add_arrays2D(wm->gate_slice, weights->ibias_gate[layer], wm->h_gate_slice, weights->hbias_gate[layer], wm->gate_gate);
            //std::cout << "Protosigmoid gate: " << std::endl << getArray2DContents(wm->gate_slice) << std::endl;
            _tanh_activation(wm->gate_gate);
            //std::cout << "Gate gate: " << std::endl << getArray2DContents(wm->gate_gate) << std::endl;

            // output gate

            add_arrays2D(wm->output_slice, weights->ibias_output[layer], wm->h_output_slice, weights->hbias_output[layer], wm->output_gate);
            //std::cout << "Protosigmoid output: " << std::endl << getArray2DContents(wm->output_gate) << std::endl;
            _sigmoid(wm->output_gate);
            //std::cout << "Output gate: " << std::endl << getArray2DContents(wm->output_gate) << std::endl;
            
            // Calc new cell state
            //std::cout << "Prev cell state: " << std::endl << getArray2DContents(prev_cell_state) << std::endl;
            hadamard_product(wm->forget_gate, wm->cell_states[layer], wm->forget_times_cell);
            hadamard_product(wm->input_gate, wm->gate_gate, wm->input_times_gate);
            add_arrays2D(wm->forget_times_cell, wm->input_times_gate, wm->cell_states[layer]);
            //std::cout << "New cell state: " << std::endl << getArray2DContents(prev_cell_state) << std::endl;

            // Calc new hidden state
            tanh_activation(wm->cell_states[layer], wm->tanh_cell);
            //fillArray2D(hidden_states[sample], 0.0);
            hadamard_product(wm->output_gate, wm->tanh_cell, wm->hidden_states[layer]);
        }

        // Copy the final layer's hidden state to output
        copyArray2D(wm->hidden_states[LSTM_NUM_LAYERS - 1], wm->output_values[sample]);
    }
    
    return wm->output_values;
}

// Sets weights to constant values.
void initializeDenoiserState(DenoiserState* ds, double sparsity, double setval)
{
  for (int i=0; i < 4 * 8; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_0_0_weight[i] = 0;
    }
    else
    {
      ds->encoder_0_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 4; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_0_0_bias[i] = 0;
    }
    else
    {
      ds->encoder_0_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 8 * 4 * 1; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_0_2_weight[i] = 0;
    }
    else
    {
      ds->encoder_0_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 8; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_0_2_bias[i] = 0;
    }
    else
    {
      ds->encoder_0_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 8 * 4 * 8; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_1_0_weight[i] = 0;
    }
    else
    {
      ds->encoder_1_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 8; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_1_0_bias[i] = 0;
    }
    else
    {
      ds->encoder_1_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 16 * 8 * 1; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_1_2_weight[i] = 0;
    }
    else
    {
      ds->encoder_1_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 16; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_1_2_bias[i] = 0;
    }
    else
    {
      ds->encoder_1_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 16 * 8 * 8; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_2_0_weight[i] = 0;
    }
    else
    {
      ds->encoder_2_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 16; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_2_0_bias[i] = 0;
    }
    else
    {
      ds->encoder_2_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 32 * 16 * 1; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_2_2_weight[i] = 0;
    }
    else
    {
      ds->encoder_2_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 32; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_2_2_bias[i] = 0;
    }
    else
    {
      ds->encoder_2_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 32 * 16 * 8; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_3_0_weight[i] = 0;
    }
    else
    {
      ds->encoder_3_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 32; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_3_0_bias[i] = 0;
    }
    else
    {
      ds->encoder_3_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 64 * 32 * 1; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_3_2_weight[i] = 0;
    }
    else
    {
      ds->encoder_3_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 64; i++) {
    if (std::rand() % 100 < sparsity)
      ds->encoder_3_2_bias[i] = 0;
    else
      ds->encoder_3_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  // Decoders
  for (int i=0; i < 64 * 32 * 1; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_0_0_weight[i] = 0;
    else
      ds->decoder_0_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 64; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_0_0_bias[i] = 0;
    else
      ds->decoder_0_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 32 * 16 * 8; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_0_2_weight[i] = 0;
    else
      ds->decoder_0_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 16; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_0_2_bias[i] = 0;
    else
      ds->decoder_0_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 32 * 16 * 1; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_1_0_weight[i] = 0;
    else
      ds->decoder_1_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 32; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_1_0_bias[i] = 0;
    else
      ds->decoder_1_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 16 * 8 * 8; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_1_2_weight[i] = 0;
    else
      ds->decoder_1_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 8; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_1_2_bias[i] = 0;
    else
      ds->decoder_1_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 16 * 8 * 1; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_2_0_weight[i] = 0;
    else
      ds->decoder_2_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 16; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_2_0_bias[i] = 0;
    else
      ds->decoder_2_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 8 * 4 * 8; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_2_2_weight[i] = 0;
    else
      ds->decoder_2_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 4; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_2_2_bias[i] = 0;
    else
      ds->decoder_2_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 8 * 4 * 1; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_3_0_weight[i] = 0;
    else
      ds->decoder_3_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 8; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_3_0_bias[i] = 0;
    else
      ds->decoder_3_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 4 * 1 * 8; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_3_2_weight[i] = 0;
    else
      ds->decoder_3_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 1; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_3_2_bias[i] = 0;
    else
      ds->decoder_3_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }
}

void randomizeWeights(double sparsity, DenoiserState* ds)
{
  // Fill weights randomly.
  std::cout << "Sparsity is " << sparsity << std::endl;
  initializeDenoiserState(ds, sparsity, -1);
}

void runDenoiser(double* inp, DenoiserState* ds, WorkingMemory* wm, double* output, LSTM_weights *lstmw, LSTM_Working_Memory *lstmwm)
{
  // Run denoiser
  //printf("Normalizing...\n");
  double SD; // Standard deviation
  if (NORMALIZE)
  {

    double sum = 0;
    int i;
    for (i=0; i < INPUT_SIZE; i++) {
        sum += inp[i];
    }
    //std::cout << "Sum: " << sum << std::endl;
    sum /= INPUT_SIZE; // This is the average.
    SD = 0;

    for (i = 0; i < INPUT_SIZE; ++i) {
        SD += pow(inp[i] - sum, 2);
    }
    // For some reason, pytorch uses N-1 instead of N.
    SD = sqrt(SD / (INPUT_SIZE - 1));

    //std::cout << "SD: " << SD << std::endl;

    // Also pad input
    for (i=0; i < VALID_LENGTH; i++) {
      wm->padded_input[i] = i < INPUT_SIZE ? (inp[i]) / (FLOOR + SD) : 0;
    }
  }
  else
  {
    int i;
    // Also pad input
    for (i=0; i < VALID_LENGTH; i++) {
      wm->padded_input[i] = i < INPUT_SIZE ? inp[i] : 0;
    }
    SD = 1; 
  }
  if (RESAMPLE != 4)
  {
    printf("RESAMPLE: %i\n", RESAMPLE);
    printf("This value is not supported. It must be 4.\n");
    exit(1);
  }

  //printf("Upsampling data...\n");
  double_upsample2_valid(wm->padded_input, wm->upsampled_input, wm);

  // Now we run each of the encoders in sequence.

  // encoder.0.0
  //printf("Encoder.0.0\n");

  int current_length = VALID_LENGTH - 1;
  conv1dChannels(wm->upsampled_input, ds->encoder_0_0_weight, ds->encoder_0_0_bias, wm->memory_grid, VALID_LENGTH * 4, current_length, KERNEL, 1, 4, STRIDE);
  
  // encoder.0.1
  //printf("Encoder.0.1\n");
  ReLU_(wm->memory_grid, current_length * 4);

  

  // encoder.0.2
  //printf("Encoder.0.2\n");
  conv1dChannels(wm->memory_grid, ds->encoder_0_2_weight, ds->encoder_0_2_bias, wm->memory_grid2, current_length, current_length, 1, 4, 8, 1);


  // encoder.0.3
  //printf("Encoder.0.3\n");
  GLU_split(wm->memory_grid2, current_length, 8, wm->memory_grid);

  
  // Copy to skips
  //printf("Copy to skips\n");
  for (int i=0; i < current_length * 4; i++) {
    wm->skip_1[i] = wm->memory_grid[i];
  }
  // encoder.1.0
  //printf("Encoder.1.0\n");
  
  conv1dChannels(wm->memory_grid, ds->encoder_1_0_weight, ds->encoder_1_0_bias, wm->memory_grid2, current_length, current_length / 4 - 1, KERNEL, 4, 8, STRIDE);
  current_length = current_length / 4 - 1;

  // encoder.1.1
  //printf("Encoder.1.1\n");
  ReLU_(wm->memory_grid2, current_length * 8);

  // encoder.1.2
  //printf("Encoder.1.2\n");
  conv1dChannels(wm->memory_grid2, ds->encoder_1_2_weight, ds->encoder_1_2_bias, wm->memory_grid, current_length, current_length, 1, 8, 16, 1);

  // encoder.1.3
  //printf("Encoder.1.3\n");
  GLU_split(wm->memory_grid, current_length, 16, wm->memory_grid2);

  // Copy to skips
  //printf("Copy to skips\n");
  for (int i=0; i < current_length * 8; i++) {
    wm->skip_2[i] = wm->memory_grid2[i];
  }

  // encoder.2.0
  //printf("Encoder.2.0\n");
  conv1dChannels(wm->memory_grid2, ds->encoder_2_0_weight, ds->encoder_2_0_bias, wm->memory_grid, current_length, current_length / 4 - 1, KERNEL, 8, 16, STRIDE);
  current_length = current_length / 4 - 1;

  // encoder.2.1
  //printf("Encoder.2.1\n");
  ReLU_(wm->memory_grid, current_length * 16);

  // encoder.2.2
  //printf("Encoder.2.2\n");
  conv1dChannels(wm->memory_grid, ds->encoder_2_2_weight, ds->encoder_2_2_bias, wm->memory_grid2, current_length, current_length, 1, 16, 32, 1);

  // encoder.2.3
  //printf("Encoder.2.3\n");
  GLU_split(wm->memory_grid2, current_length, 32, wm->memory_grid);

  // Copy to skips
  //printf("Copy to skips\n");
  for (int i=0; i < current_length * 16; i++) {
    wm->skip_3[i] = wm->memory_grid[i];
  }
  
  // encoder.3.0
  //printf("Encoder.3.0\n");
  conv1dChannels(wm->memory_grid, ds->encoder_3_0_weight, ds->encoder_3_0_bias, wm->memory_grid2, current_length, current_length / 4 - 1, KERNEL, 16, 32, STRIDE);
  current_length = current_length / 4 - 1;

  // encoder.3.1
  ReLU_(wm->memory_grid2, current_length * 32);

  // encoder.3.2
  conv1dChannels(wm->memory_grid2, ds->encoder_3_2_weight, ds->encoder_3_2_bias, wm->memory_grid, current_length, current_length, 1, 32, 64, 1);
    
  // encoder.3.3
  GLU_split(wm->memory_grid, current_length, 64, wm->memory_grid2);

  // Copy to skips
  for (int i=0; i < current_length * 32; i++) {
    wm->skip_4[i] = wm->memory_grid2[i];
  }

  //printf("Run LSTM\n");
  // Copy input into LSTM input
  // It currently is [LSTM_num_samples x 1 x lstm_input_size]
  // Need it to be [LSTM_num_samples x lstm_input_size x 1]
  
  
  for (int j = 0; j < current_length; j++)
  {
    for (int i = 0; i < 32; i++)
    {
      set(lstmwm->input_sequence[j], i, 0, wm->memory_grid2[j + i * current_length]);
    }
  }
  
  Array2D** lstm_out = LSTM_forward(lstmw, lstmwm);

  // Now copy it back.

  for (int j = 0; j < current_length; j++)
  {
    for (int i=0; i < 32; i++) 
    {
      wm->memory_grid2[i * current_length + j] = get(lstm_out[j], i, 0);
    }
  }

  // Add skip4
  for (int i=0; i < current_length * 32; i++) {
    wm->memory_grid2[i] += wm->skip_4[i];
  }
  
  // decoder.0.0
  //printf("Decoder.0.0\n");
  conv1dChannels(wm->memory_grid2, ds->decoder_0_0_weight, ds->decoder_0_0_bias, wm->memory_grid, current_length, current_length, 1, 32, 64, 1);

  // encoder.0.1
  //printf("Decoder.0.1\n");
  GLU_split(wm->memory_grid, current_length, 64, wm->memory_grid2);  



  // decoder.0.2
  //printf("Decoder.0.2\n");
  conv1dTransposeChannels(wm->memory_grid2, ds->decoder_0_2_weight, ds->decoder_0_2_bias, wm->memory_grid, current_length, KERNEL, 32, 16, STRIDE);
  current_length = (current_length - 1) * STRIDE + KERNEL;

  // decoder.0.3
  //printf("Decoder.0.3\n");
  ReLU_(wm->memory_grid, current_length * 16);

  // Add skip3
  for (int i=0; i < current_length * 16; i++) {
    wm->memory_grid[i] += wm->skip_3[i];
  }

  // decoder.1.0
  //printf("Decoder.1.0\n");
  conv1dChannels(wm->memory_grid, ds->decoder_1_0_weight, ds->decoder_1_0_bias, wm->memory_grid2, current_length, current_length, 1, 16, 32, 1);
  
  // encoder.1.1
  //printf("Decoder.1.1\n");
  GLU_split(wm->memory_grid2, current_length, 32, wm->memory_grid);

  // decoder.1.2
  //printf("Decoder.1.2\n");
  conv1dTransposeChannels(wm->memory_grid, ds->decoder_1_2_weight, ds->decoder_1_2_bias, wm->memory_grid2, current_length, KERNEL, 16, 8, STRIDE);
  current_length = (current_length - 1) * STRIDE + KERNEL;

  // decoder.1.3
  //printf("Decoder.1.3\n");
  ReLU_(wm->memory_grid2, current_length * 8);

  // Add skip2
  for (int i=0; i < current_length * 8; i++) {
    wm->memory_grid2[i] += wm->skip_2[i];
  }

  // decoder.2.0
  //printf("Decoder.2.0\n");
  conv1dChannels(wm->memory_grid2, ds->decoder_2_0_weight, ds->decoder_2_0_bias, wm->memory_grid, current_length, current_length, 1, 8, 16, 1);

  // decoder.2.1
  //printf("Decoder.2.1\n");
  GLU_split(wm->memory_grid, current_length, 16, wm->memory_grid2);

  // decoder.2.2
  //printf("Decoder.2.2\n");
  conv1dTransposeChannels(wm->memory_grid2, ds->decoder_2_2_weight, ds->decoder_2_2_bias, wm->memory_grid, current_length, KERNEL, 8, 4, STRIDE);
  current_length = (current_length - 1) * STRIDE + KERNEL;

  // decoder.2.3
  //printf("Decoder.2.3\n");
  ReLU_(wm->memory_grid, current_length * 4);
  

  // Add skip1
  for (int i=0; i < current_length * 4; i++) {
    wm->memory_grid[i] += wm->skip_1[i];
  }

  // decoder.3.0
  //printf("Decoder.3.0\n");
  conv1dChannels(wm->memory_grid, ds->decoder_3_0_weight, ds->decoder_3_0_bias, wm->memory_grid2, current_length, current_length, 1, 4, 8, 1);
    

  // decoder.3.1
  //printf("Decoder.2.1\n");
  GLU_split(wm->memory_grid2, current_length, 8, wm->memory_grid);

  
  // decoder.3.2
  //printf("Decoder.3.2\n");
  conv1dTransposeChannels(wm->memory_grid, ds->decoder_3_2_weight, ds->decoder_3_2_bias, wm->memory_grid2, current_length, KERNEL, 4, 1, STRIDE);
  current_length = (current_length - 1) * STRIDE + KERNEL;

  if (RESAMPLE == 4)
  {
    downsample2_consts(wm->memory_grid2, wm->memory_grid, wm, current_length);
    downsample2_consts(wm->memory_grid, wm->memory_grid2, wm, current_length / 2);
  }

  for (int i=0; i < INPUT_SIZE; i++) {
    output[i] = wm->memory_grid2[i] * (SD);
  }
  //print_array(output, 0, 100, "Memory Grid");
  //std::cout << "Current length: " << current_length << std::endl;
  //exit(1);
}

// Loads weights to fill the denoiser state.
void loadWeightsFromDisk(const char* filename, DenoiserState* ds, LSTM_weights *lstm_weights)
{
  std::cout << "Loading weight file" << std::endl;
  //FILE* file = fopen(filename, "r");

  std::vector<std::string> tokens;
  std::stringstream check(WEIGHTS);
  std::string intermediate;
  while (getline(check, intermediate, ' '))
  {
      tokens.push_back(intermediate);
  }

  //const char* WEIGHTS = "0.07480469 0.05065702 -0.02524773 -0.11673886 -0.02747198 -0.24519193 -0.00017723 0.14770360 -0.62041670 -0.51342577 -0.41662985 0.33385754 0.53671110 0.63397241 0.18184608 -0.08239134 0.57240903 0.80008805 0.35545000 -0.42137986 -0.67511022 -0.51229745 -0.26452038 0.08775014 -0.04732119 -0.25180668 -0.21434259 0.40078744 0.37893158 -0.27435738 -0.41630173 0.32849538";
  for (int i = 0; i < 4 * 8; i++)
  {
    ds->encoder_0_0_weight[i] = std::stod(tokens.front(), nullptr);
    tokens.erase(tokens.begin());
  }
  //WEIGHTS = "0.66450578 -0.19182815 0.21062097 1.30057836";

  for (int i = 0; i < 4; i++)
  {
      ds->encoder_0_0_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  //WEIGHTS = "0.35157460 -0.23550436 0.20619105 -0.09416435 -0.08865921 1.20109200 -1.08440065 0.23584421 0.34043869 -1.18597853 1.08113432 -0.66227019 -0.02683747 -0.32499382 0.38166443 -0.98510253 0.03804530 -0.00149668 0.01239207 0.16828743 -0.65140724 -0.24740300 4.73097229 1.83127940 0.01353258 0.00795421 -0.05847628 -0.31024623 -0.38490453 0.19337897 0.16443051 0.81370342";

  for (int i = 0; i < 4 * 8; i++)
  {
      ds->encoder_0_2_weight[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  //WEIGHTS = "-0.31428379 -0.25246748 0.24959418 -0.57360017 -0.27694887 2.18235207 0.65860879 0.93812978";

  for (int i = 0; i < 8; i++)
  {
      ds->encoder_0_2_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  //WEIGHTS = "0.04110954 -0.24435110 0.23870996 -0.18419108 0.07715700 0.18232164 -0.61442542 0.13114168 0.14189884 -0.11597596 0.05656343 -0.28905082 0.12952976 -0.06579456 0.02943041 -0.07731058 -0.13101570 0.04978158 -0.30865839 0.39188334 -0.52940750 0.46995428 -0.28921854 0.22231011 -0.09850591 -0.06244304 -0.47971925 0.32815063 -0.27717775 -0.01201211 0.00244374 -0.22695583 -0.93778569 -0.10458615 -0.39722204 -0.70780587 -0.84717143 -1.47149622 -1.03273714 -1.58794570 -0.08855549 -0.13391219 -0.03230653 0.15418744 -0.17084709 -0.28802294 -0.15226997 -0.16253783 -0.37317488 -0.67671019 -0.56378287 0.15617162 0.33760113 0.37658554 -0.02229470 -0.11920354 0.02444298 -0.09207815 -0.08310385 0.29682589 0.06106063 0.35472080 0.56508982 0.18290497 0.67064977 0.69892430 0.70615727 1.18020701 1.45869839 0.95012206 0.45315817 0.90413857 -0.03849063 0.04196979 -0.04236987 0.01736802 0.16257410 0.23453385 0.05317980 0.01849129 0.19394769 0.38597572 0.49007937 -0.10035551 -0.29222670 -0.26805180 -0.19181769 0.00328787 -0.09628076 0.05563187 -0.08888439 -0.09494615 -0.09370754 -0.36221659 -0.16398174 -0.20475888 0.93704093 0.61208409 -0.34507835 0.03745240 0.45457038 0.21922882 0.21238196 0.39933026 -0.20123981 -0.15759555 -0.48424587 -0.70014912 -0.31332895 -0.22427872 -0.23542656 0.11986490 -0.31569818 0.23621786 0.20718548 0.68089467 0.70640981 0.68979347 -0.17177565 -0.09644496 -0.02813371 0.20130910 0.28076664 0.14671160 0.08811622 0.25990498 -0.10079860 -0.09345528 -0.71556246 -0.52981180 -0.28237334 -0.31956151 -0.06712109 -0.67251462 -0.28468966 -0.64864790 0.00075917 0.35040265 0.48647559 0.89224303 0.53185785 0.55497360 0.17576930 0.03488391 -0.01851452 -0.11758731 -0.36774495 -0.59881312 -0.57525516 -0.35074359 0.13925245 0.28969702 -0.19858479 -0.16909155 -0.11393046 -0.35219637 -0.22989826 -0.16939963 -0.03565443 0.29432112 -0.51154137 -0.44503739 -0.47738221 -0.48610240 -0.03743584 -0.10624996 -0.32197952 0.05783274 -0.43229508 -0.30277616 -0.06980536 0.26860151 0.04127390 -0.12969071 -0.18412149 -0.05171507 0.58017069 0.21329078 0.02131158 -0.28451800 0.08891790 -0.02416802 -0.08644893 -0.12350288 -0.14899094 -0.06567683 -0.31445321 -0.20413396 -0.34432769 -0.24790215 -0.23183072 -0.39026275 0.11516244 -0.29166737 -0.08490113 -0.63450801 -0.40849596 -1.12867105 -0.51634419 -0.32032898 -0.03881367 0.11594897 -0.29776293 -0.36070552 -0.45444307 0.13624814 -0.25807151 -0.31668180 -0.52253795 -0.54052901 0.34235463 -0.33289340 0.37939486 -0.33636877 -0.03497174 0.01078758 -0.35176578 -0.01189746 0.07463165 -0.10560062 -0.23799440 -0.39963481 -0.18287042 -0.39145744 0.60843211 -0.74520981 0.16666996 -0.29833442 -0.02799881 -1.03448355 0.20250459 -0.13634200 -0.29729500 -0.17790754 0.40046260 0.56041092 0.01812090 -0.54886305 -0.07696787 0.27107522 0.74937898 0.16723566 -0.44465321 -0.73261547 0.36514008 0.97113580 0.50131786 -0.74565518 -0.20220910 -0.04461214 -0.38263139 0.37667587 0.35051492 0.36909658 -0.15363163 -0.38800749";

  for (int i = 0; i < 8 * 4 * 8; i++)
  {
      ds->encoder_1_0_weight[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  //WEIGHTS = "0.24812469 0.45862764 -0.28631043 0.26373786 -0.27379534 -0.01336762 0.25175565 -0.10995601";
  for (int i = 0; i < 8; i++)
  {
      ds->encoder_1_0_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  //WEIGHTS = "-0.06641925 0.54013729 -0.56667352 -0.14230286 0.14599299 0.11004222 -0.11627874 0.03806527 -0.12263034 -0.29128045 0.46858677 -0.15822656 0.18754503 0.40210110 -0.33827147 0.08932289 -0.09746283 -0.60678202 0.43180263 -0.34466505 0.20275496 0.10455562 -0.31869426 0.08732741 2.27480197 0.28903723 -0.35436073 0.03052988 -0.01797874 -0.00012470 -0.79171604 0.11782892 0.66386515 -0.01126262 -0.02316223 0.37505415 -0.40100515 -0.27643731 0.08802543 0.56451041 1.08065557 0.28178039 0.01574873 -0.31487423 0.46694350 -0.53886318 -0.02329859 0.75370634 0.16366601 -0.03564592 -0.06062725 -0.34575641 0.32147023 -0.06182855 0.04539280 0.04834555 -1.41373003 -0.30217621 0.26221892 -0.09675252 0.08440578 0.27916995 0.20563290 0.21306249 0.09162863 0.07229216 -0.08145390 0.02134944 -0.01720706 0.10233295 0.03184394 -0.02804301 0.09918189 -0.02479556 -0.01725014 -0.04273577 0.02821973 0.36535883 0.26873496 -0.01423066 0.06504958 -0.00394880 -0.03113072 0.03082242 -0.01552491 0.04083700 0.14124127 0.03394248 0.06416281 0.27440563 -0.48956445 -0.01161558 -0.04880171 0.58862263 0.48903418 -0.12495998 -0.01999391 0.11915235 -0.15917569 0.07413135 -0.07529841 0.08930462 0.11957074 -0.08187144 -0.18592612 -0.00729170 -0.09214549 -0.01063697 -0.01175412 -0.14246114 -0.13160154 -0.04511811 0.19041198 0.16401714 0.14772385 0.81297225 0.02018954 0.21085431 0.04115675 -0.02959003 0.00972164 0.10873689 -0.13390067 0.04010737 -0.00846841 0.17442437 0.25911340 -0.04686545";

  for (int i = 0; i < 16 * 8; i++)
  {
      ds->encoder_1_2_weight[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  //WEIGHTS = "0.36754480 -0.46150780 0.38673916 -0.72120363 -0.01977813 -0.13031289 -0.48360631 0.37879914 1.09036529 -0.50717252 -0.86008167 -0.10327384 0.54968518 -0.11018800 0.28885978 0.29300627";

  for (int i = 0; i < 16; i++)
  {
      ds->encoder_1_2_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  //WEIGHTS = "0.18211274 0.32044363 -0.02743929 0.25978583 0.40275311 -0.10259899 -0.19901979 0.02459831 0.00532619 0.03119784 0.12654346 -0.06235534 -0.15638247 -0.04483924 0.02528092 -0.00963523 -0.16042365 -0.02953659 -0.57258570 -0.17277184 -0.04476227 -0.18507479 -0.10825338 -0.10202409 0.00273496 -0.16536608 0.01357406 -0.00472119 -0.07583508 0.03342918 0.10984150 0.02574258 0.16775270 0.45027184 0.25077522 0.32717758 0.43787837 0.28453219 0.02693806 0.02353192 0.04516758 0.36694017 0.62655771 0.67652309 0.70782065 0.60133159 0.48546076 0.21845827 -0.28622785 -0.20683460 0.37294626 -0.00539399 -0.23635709 0.34043172 0.33241314 0.09670741 0.10202560 -0.34378186 -0.11318545 -0.01210519 -0.16448897 -0.22071828 -0.06266901 0.07167649 0.04633979 0.14450905 -0.35573342 -0.54378796 -0.01987075 0.24000922 0.26481959 0.04134041 -0.04004331 -0.05893409 0.09668970 0.24873048 0.09592970 0.01366857 -0.17764348 -0.18895636 -0.09730817 0.05347939 0.10775249 -0.17637976 -0.02607147 0.01841547 -0.04211218 0.08332416 0.02624189 -0.07417285 -0.11648776 0.15170740 0.17787942 -0.00740268 -0.00123375 -0.07186967 0.02863449 0.15017423 -0.01063887 -0.30067575 -0.18486784 0.11408635 0.19728832 0.16243225 -0.05001330 -0.03058842 0.35043284 0.20409159 -0.03727299 -0.09079400 -0.21394247 -0.12657979 -0.28342363 -0.33882678 0.64511710 1.03546202 0.18483844 -0.50621146 -0.54516464 0.11569361 0.08210415 -0.10167766 -0.23532082 0.21578056 0.30376124 0.03783125 0.07446871 -0.14812116 -0.12126349 -0.34255439 -0.32175410 0.15394454 0.14512290 -0.21158126 0.02156982 0.03791399 0.07925023 0.20568037 0.19781360 0.02607810 -0.06992084 -0.07271267 -0.03478491 -0.06622235 0.00832973 0.02672662 -0.29926628 -0.08508995 0.02343725 0.07167327 0.13033012 0.02299840 -0.05156016 0.01264813 0.20248213 0.09967068 -0.03564256 -0.03532260 -0.03515372 -0.03631430 -0.03266773 -0.13757385 -0.30713004 -0.07197114 0.19133748 0.19852611 0.18100148 0.05195453 0.23057416 0.21895914 0.20182517 0.00792411 -0.08665034 -0.20686375 -0.43669933 -0.11262574 0.37721056 0.85539818 0.57770449 -0.32620424 -0.75018609 -0.47643948 -0.30971846 -0.02951022 -0.09393145 -0.03246704 0.33875757 0.19822378 -0.03051918 -0.06008612 -0.05364123 -0.08610880 -0.02419446 0.43388250 -0.60105801 0.33433816 0.05947205 -0.49454230 0.11321162 0.28289124 -0.22183707 0.09531672 -0.11221142 0.26860705 -0.75378180 0.41558695 0.06423612 0.18331245 0.41717902 -0.77348304 0.46857220 0.34844303 0.55691153 -0.10745285 -0.24914807 -0.34026709 0.07019634 0.04058085 -0.14283726 0.09481527 0.01015250 0.06493701 -0.02188077 -0.10510076 -0.06834232 -0.13563229 0.38212651 -0.50083280 0.22793043 -0.04439066 -0.00422291 0.07119163 0.00325443 0.06777462 -0.04629777 0.17337868 -0.21431872 -0.18965113 -0.00657877 -0.11336444 -0.15432735 -0.23460479 0.27785668 0.24062899 -0.27399182 -0.12695865 0.23188651 0.02672333 0.13562784 0.06670850 -0.21846783 0.10704671 0.09248164 0.07410643 -0.04468036 -0.17166889 -0.13935594 -0.00795425 0.06349342 -0.05775648 -0.04512222 -0.03082364 -0.19779007 -0.21138112 0.14968020 0.13569091 0.27339631 0.09987301 -0.12746307 -0.09980270 -0.07505730 -0.12133507 -0.23790176 -0.28327921 -0.55186307 -0.21132797 -0.00537002 0.00813684 0.06803331 -0.04521741 0.03841935 0.08333008 0.09577045 0.03187970 0.12195215 0.27428898 0.22168605 0.09495883 0.10083523 0.34881833 0.54448479 0.52659285 0.47123951 0.40778768 0.13029233 0.04083779 0.31756505 0.33123642 0.56083083 0.90446907 0.88942105 0.80675310 0.83687788 0.48135063 0.21964447 0.20827682 0.10660529 0.14941296 0.30843329 0.14254604 -0.01211178 0.16167603 0.09451250 -0.09022219 -0.31421027 -0.22905992 -0.04001073 -0.21815857 -0.22723988 0.13481979 0.04042307 -0.02858062 -0.02266307 -0.16631041 -0.08247020 0.20061959 0.09672514 0.06986571 0.11450675 0.15456037 0.35393962 0.07743128 -0.09627254 -0.02495302 -0.01395492 0.00831945 -0.16946617 -0.14359392 -0.55713159 -0.33646062 -0.21140875 -0.03565558 0.07633144 -0.01991852 -0.18229413 -0.49975461 -0.62714028 -0.76876640 -0.65963566 -0.51852626 -0.36415619 -0.14079598 0.12571545 0.23268187 0.27165025 0.20089938 0.18884200 0.26073018 0.11676499 0.07398947 0.21505293 0.54146576 0.53823388 0.78664523 0.56520307 0.39192721 0.42758447 0.19584599 0.13129950 0.41997755 0.39736262 0.40454131 0.27792114 0.05487930 0.00064166 -0.03976674 0.08978618 -0.00755482 0.27761015 0.63364899 0.90915465 0.46775174 0.15770340 0.18787104 0.01523725 0.21912126 0.06463818 -0.69166321 0.66604173 -0.03542479 -0.39514324 0.19582818 -0.02552417 -0.03611737 0.20307276 0.04196395 0.08961888 -0.78854543 0.70688814 -0.20128122 0.10077800 0.04448991 -0.72992009 0.90676671 -0.43498030 1.14136565 -0.73382455 0.07386049 0.02804430 -0.04335893 -0.01804474 -0.08855157 0.12413623 -0.04432285 0.02768119 0.02824477 -0.03326983 -0.00277646 0.05162987 0.12853472 -0.32528767 0.28264794 -0.20907389 0.08240150 -0.16913281 0.16000372 -0.05862806 0.17683619 0.01767815 -0.33792773 0.01488578 -0.16459560 0.14394662 -0.06318898 -0.16116866 0.09357916 0.31251177 -0.35660470 -0.07819866 0.21608895 0.05636014 -0.06380749 -0.05278881 -0.13349453 0.22337951 -0.05239013 -0.04006556 0.11680454 -0.08101144 0.10795256 0.07662570 -0.00709476 -0.08371871 -0.23632699 -0.51070642 -0.21742703 -0.09695353 -0.10696961 -0.07014168 0.03273030 -0.06883597 0.04702483 0.28383893 0.26882854 -0.05479725 0.10148837 0.06882386 -0.07293139 0.17582452 0.28489420 0.37455305 0.20228748 0.11232729 -0.01724799 -0.06293147 0.00180925 0.06108472 -0.03994882 -0.02026587 0.17644978 0.13467669 0.17562313 0.10926773 0.00116576 -0.10108782 -0.08569038 -0.27692223 -0.22872597 -0.47864640 -0.36171359 0.02492217 0.02883229 0.09892709 -0.09140756 -0.11154369 -0.21731013 -0.05963617 -0.08624944 -0.01848723 0.16124766 0.26734462 0.60601455 0.58659369 -0.14992629 0.23501033 -0.02480633 -0.13334566 -0.03898580 -0.02526527 -0.09497727 -0.11619060 0.25210431 -0.04144203 0.00537683 0.62767410 0.39103004 -0.18582258 -0.32364550 -0.33321711 -0.03419948 0.07334201 -0.03579587 -0.09855943 -0.23093496 -0.14865832 0.03421262 0.30014318 0.26253113 0.11911567 -0.10411398 -0.06078383 0.09334815 0.01116169 -0.04672758 0.05260543 -0.14766969 -0.06156568 0.15174988 0.00283420 -0.16589621 -0.06156585 0.00745929 -0.01442658 0.12619424 -0.06266287 -0.25453925 0.04732325 0.43779591 0.11898655 -0.22534207 -0.29201394 -0.19255395 0.26915252 0.19211696 -0.07311669 -0.40518415 -0.01476941 0.36743832 0.33672348 -0.20470667 0.31673786 0.22306599 -0.74670810 -0.80417448 -0.31214792 0.48764190 0.65433711 -0.05144985 -0.11348264 0.22534887 0.06182776 -0.21184593 -0.08809457 -0.03831137 -0.15858716 0.21202481 0.02541180 -0.13607687 0.19549233 0.18504865 -0.56131995 -0.32865205 -0.10324517 -0.11007026 0.10445551 0.11395396 0.17163551 0.01118579 0.06640575 -0.10764540 -0.19370788 0.03367573 -0.23163396 -0.40958631 -0.77169228 -0.32530022 -0.05246982 -0.20610349 -0.06126962 0.21978258 -0.04235511 -0.07325040 -0.06494703 -0.27219301 -0.01178149 -0.00300161 -0.04342175 -0.02478789 0.28706560 0.33613950 0.33170807 0.59186548 0.38661277 0.24240971 0.22818096 0.16073960 -0.11316825 0.54473805 1.09377539 1.58930385 1.45864463 0.71087396 0.06290598 -0.03103274 -0.05190229 0.17753656 0.12915879 0.13096988 0.41123506 0.51814955 0.32702330 0.17854382 0.08826617 0.03202204 0.08911610 -0.26267824 -0.03230943 0.34981105 0.28771538 -0.07848769 0.15470722 -0.11733785 -0.11021644 -0.09331667 0.03356517 0.14905812 0.35903075 0.15494576 -0.09125969 -0.10167718 -0.18273461 -0.03329222 0.17909661 0.07642481 -0.08211821 -0.05213456 0.56880260 0.58127904 0.54116040 0.03640866 -0.50392175 -0.57589567 -0.34661442 -0.19103904 -0.12928185 0.03090246 0.14398177 0.06340799 -0.01681340 -0.02191208 -0.08460800 -0.16871209 -0.02892806 -0.05655365 -0.16027938 -0.14584740 -0.05229555 -0.02623078 0.15406567 0.19828133 -0.09687283 -0.64567822 -0.56595314 -0.02170580 0.45365593 0.88064891 0.61101002 0.15029752 0.15282680 0.40244043 -0.04972488 -0.19992878 -0.44881213 -0.67754054 -0.60291195 0.09447561 -0.27011815 -0.02249676 0.20539652 0.10621855 0.07168341 0.03877410 -0.05047330 -0.27647358 -0.08910009 -0.15931401 -0.24847999 -0.40022835 -0.55975133 -0.76683265 -0.48093513 -0.14694650 -0.01547727 0.04190489 0.04706744 -0.00971145 0.02281347 0.12264048 -0.00471429 -0.06603098 -0.00518809 -0.05387451 0.02336636 -0.03584315 0.20605910 0.13048309 0.01826983 0.05029999 0.01615164 0.03345584 -0.03354064 0.01643870 -0.09005187 -0.01765487 0.11077289 -0.05325959 -0.06868954 0.06416959 0.19651563 0.25830874 0.39228085 0.26211753 0.13871095 0.12995367 -0.06761079 -0.23834485 -0.30137441 -0.38356009 -0.35567555 -0.22147320 -0.21429825 0.01721778 -0.12428411 -0.55379295 -0.73126578 -0.87426299 -0.73155433 -0.49699688 -0.51155454 -0.05112642 -0.00667321 0.05367786 -0.08437705 0.05374876 -0.12093923 -0.05204315 0.24473901 -0.06321595 -0.09073743 0.28762385 0.29641101 -0.77331573 0.48333409 0.04431547 -0.32489312 0.19358960 0.04279234 0.00355299 -0.12029441 0.46010444 -0.24424380 -0.09430997 0.07060558 -0.11391234 -0.08828854 0.11012948 0.06688832 0.41333136 -0.17556420 -0.18004876 0.09603783 -0.07464580 0.02093181 -0.19907284 -0.29212698 0.64795280 -0.53445792 0.19796862 0.38821420 -0.19576657 -0.00080613 0.02955638 -0.05599659 -0.04039859 -0.03036406 -0.04477117 -0.01619258 -0.01537232 -0.06456158 -0.08637784 0.23595257 0.22116315 -0.32805333 -0.14266714 0.23532824 -0.00661783 0.11845090 -0.13462262 -0.06527242 -0.13290909 0.01777495 -0.18065664 -0.03282502 0.03951653 0.00012774 0.00006051 0.00101812 -0.64514023 0.54438406 0.18675277 -0.29885122 0.14166906 -0.23957317 -0.24270576 -0.41624504 0.98433262 0.60484076 -0.01225136 -0.31241626 -0.10090310 0.22722654 -0.05714617 0.11484590 -0.21402565 -0.17148285 0.08477487 0.08239715 -0.01510642 0.10033627 -0.08405135 -0.15566073 -0.16582955 0.06602264 0.12827669 0.11787692 -0.03446489 0.39107037 -0.09206642 -0.14908303 -0.67864567 -0.29453665 0.43369567 0.39469755 0.07211293 0.00689356 -0.04953267 -0.17367320 -0.00149398 0.10693047 0.05746365 0.05016068 -0.00585969 -0.14907162 -0.04562371 -0.10961942 -0.58708435 0.28242427 0.13871568 -0.04661558 -0.08298945 -0.17973424 0.28038025 0.19169204 -0.60388011 -0.21731724 -0.14345889 0.25409091 0.12027915 -0.36117738 0.11110900 0.38045865 0.74094558 -0.14996845 -0.45838085 -0.41085079 0.06554126 0.19250806 0.46850780 0.11728390 -0.22088282 -0.10765652 0.23173977 0.06065702 -0.00007797 -0.13101636 -0.15641174 -0.17056932 0.02961915 0.19462912 0.05779256 0.07837627 0.05749350 0.00226913 0.06507576 0.11330198 0.25408763 0.01671886 -0.14880770 -0.17410952 -0.09678655 0.07132173 -0.11371277 -0.14355531 -0.12710166 -0.03500938 0.04062758 -0.00929562 0.06293748 0.03312727 0.25998509 0.25720522 0.05567341 -0.14130501 -0.17935507 -0.10576244 -0.08272068 -0.20423698 -0.30138406 -0.15444790 0.07686859 0.17245324 0.16338247 0.22453997 0.00734597 -0.48084790 -0.71519899 -0.37140140 0.20083970 0.57036132 0.40419993 0.39209735 0.02187461 0.16030207 -0.14135215 -0.23016569 -0.22257493 -0.06051559 0.08877034 -0.04823458 0.11720448 0.12223186 0.23882627 0.30695009 0.29578903 0.34660795 0.58298481 0.40533110 0.03105235 -0.01457383 -0.03927173 -0.08012120 -0.08809330 -0.09100559 -0.06876424 0.06513371 0.09670557 0.10433885 0.15130560 0.11918484 0.12810108 -0.13948709 -0.03343400 -0.04475671 -0.21847799 -0.02539776 -0.06512991 -0.01288391 0.01807695 0.12902194 -0.00128219 -0.11328606 0.03443060 0.05498343 -0.00629017 -0.09389099 -0.18566729 -0.35339785 -0.22628307 -0.08803219 -0.11890960 0.01085217 0.12908946 0.10281987 0.20079109 0.27495712 0.22787285 0.23704588 0.30026796 0.09624854 0.41768780 0.65883708 0.80226547 0.63435185 0.39964792 0.41620299 0.05351699 -0.02906726 -0.09486186 -0.00762823 0.02209914 0.20127504 0.02465927 -0.21298677 0.07316127";
  for (int i = 0; i < 16 * 8 * 8; i++)
  {
      ds->encoder_2_0_weight[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  //WEIGHTS = "-0.00751677 0.02207307 0.01602714 0.01270574 -0.00031822 0.30928326 0.01896110 0.28865457 -0.00511396 0.01049182 -0.14889275 0.00643702 -0.00031455 0.01925462 0.00326359 0.00519888";

  for (int i = 0; i < 16; i++)
  {
      ds->encoder_2_0_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }



  for (int i = 0; i < 32 * 16; i++)
  {
      ds->encoder_2_2_weight[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }
  for (int i = 0; i < 32; i++)
  {
      ds->encoder_2_2_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  for (int i = 0; i < 32 * 16 * 8; i++)
  {
      ds->encoder_3_0_weight[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }
  for (int i = 0; i < 32; i++)
  {
      ds->encoder_3_0_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  for (int i = 0; i < 64 * 32; i++)
  {
      ds->encoder_3_2_weight[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }
  for (int i = 0; i < 64; i++)
  {
      ds->encoder_3_2_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  
  for (int i = 0; i < 64 * 32 * 1; i++)
  {
      ds->decoder_0_0_weight[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }
  for (int i = 0; i < 64; i++)
  {
      ds->decoder_0_0_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  for (int i = 0; i < 32 * 16 * 8; i++)
  {
      ds->decoder_0_2_weight[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }
  for (int i = 0; i < 16; i++)
  {
      ds->decoder_0_2_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }


  for (int i = 0; i < 32 * 16 * 1; i++)
  {
      ds->decoder_1_0_weight[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }
  for (int i = 0; i < 32; i++)
  {
      ds->decoder_1_0_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  for (int i = 0; i < 16 * 8 * 8; i++)
  {
      ds->decoder_1_2_weight[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }
  for (int i = 0; i < 8; i++)
  {
      ds->decoder_1_2_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }


  for (int i = 0; i < 16 * 8 * 1; i++)
  {
      ds->decoder_2_0_weight[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }
  for (int i = 0; i < 16; i++)
  {
      ds->decoder_2_0_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  for (int i = 0; i < 8 * 4 * 8; i++)
  {
      ds->decoder_2_2_weight[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }
  for (int i = 0; i < 4; i++)
  {
      ds->decoder_2_2_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  
  for (int i = 0; i < 8 * 4 * 1; i++)
  {
      ds->decoder_3_0_weight[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }
  for (int i = 0; i < 8; i++)
  {
      ds->decoder_3_0_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }

  for (int i = 0; i < 4 * 1 * 8; i++)
  {
      ds->decoder_3_2_weight[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }
  for (int i = 0; i < 1; i++)
  {
      ds->decoder_3_2_bias[i] = std::stod(tokens.front(), nullptr);
      tokens.erase(tokens.begin());
  }
  
  // Now load LSTM weights
  for (int i = 0; i < LSTM_NUM_LAYERS; i++)
  {
    for (int j = 0; j < 4 * LSTM_HIDDEN_SIZE * LSTM_INPUT_SIZE; j++)
    {
        lstm_weights->weight_ih[i]->data[j] = std::stod(tokens.front(), nullptr);
        tokens.erase(tokens.begin());
      //sscanf(WEIGHTS, "%lf", &lstm_weights->weight_ih[i]->data[j]);
    }
    for (int j = 0; j < 4 * LSTM_HIDDEN_SIZE * LSTM_INPUT_SIZE; j++)
    {
        lstm_weights->weight_hh[i]->data[j] = std::stod(tokens.front(), nullptr);
        tokens.erase(tokens.begin());
      //sscanf(WEIGHTS, "%lf", &lstm_weights->weight_hh[i]->data[j]);
    }

    // Biases
    for (int j = 0; j < LSTM_HIDDEN_SIZE; j++)
    {
        lstm_weights->ibias_input[i]->data[j] = std::stod(tokens.front(), nullptr);
        tokens.erase(tokens.begin());
      //sscanf(WEIGHTS, "%lf", &lstm_weights->ibias_input[i]->data[j]);
    }
    for (int j = 0; j < LSTM_HIDDEN_SIZE; j++)
    {
        lstm_weights->ibias_forget[i]->data[j] = std::stod(tokens.front(), nullptr);
        tokens.erase(tokens.begin());
      //sscanf(WEIGHTS, "%lf", &lstm_weights->ibias_forget[i]->data[j]);
    }
    for (int j = 0; j < LSTM_HIDDEN_SIZE; j++)
    {
        lstm_weights->ibias_gate[i]->data[j] = std::stod(tokens.front(), nullptr);
        tokens.erase(tokens.begin());
      //sscanf(WEIGHTS, "%lf", &lstm_weights->ibias_gate[i]->data[j]);
    }
    for (int j = 0; j < LSTM_HIDDEN_SIZE; j++)
    {
        lstm_weights->ibias_output[i]->data[j] = std::stod(tokens.front(), nullptr);
        tokens.erase(tokens.begin());
      //sscanf(WEIGHTS, "%lf", &lstm_weights->ibias_output[i]->data[j]);
    }

    // Hidden biases
    for (int j = 0; j < LSTM_HIDDEN_SIZE; j++)
    {
        lstm_weights->hbias_input[i]->data[j] = std::stod(tokens.front(), nullptr);
        tokens.erase(tokens.begin());
      //sscanf(WEIGHTS, "%lf", &lstm_weights->hbias_input[i]->data[j]);
    }
    for (int j = 0; j < LSTM_HIDDEN_SIZE; j++)
    {
        lstm_weights->hbias_forget[i]->data[j] = std::stod(tokens.front(), nullptr);
        tokens.erase(tokens.begin());
      //sscanf(WEIGHTS, "%lf", &lstm_weights->hbias_forget[i]->data[j]);
    }
    for (int j = 0; j < LSTM_HIDDEN_SIZE; j++)
    {
        lstm_weights->hbias_gate[i]->data[j] = std::stod(tokens.front(), nullptr);
        tokens.erase(tokens.begin());
      //sscanf(WEIGHTS, "%lf", &lstm_weights->hbias_gate[i]->data[j]);
    }
    for (int j = 0; j < LSTM_HIDDEN_SIZE; j++)
    {
        lstm_weights->hbias_output[i]->data[j] = std::stod(tokens.front(), nullptr);
        tokens.erase(tokens.begin());
      //sscanf(WEIGHTS, "%lf", &lstm_weights->hbias_output[i]->data[j]);
    }
    
  }
  

  //fclose(file);
  
}

void initializeWorkingMemory(WorkingMemory* wm)
{  
    int i;
    std::cout << "Initializing working memory" << std::endl;

    for (i=0; i < VALID_LENGTH; i++) {
      wm->padded_input[i] = 0;
      wm->upsample_working[i] = 0;
    }

    for (i=0; i < 2 * VALID_LENGTH; i++) {
      wm->upsample_working_double[i] = 0;
      wm->half_input_one[i] = 0;
      wm->half_input_two[i] = 0;
    }

    for (i = 0; i < VALID_LENGTH + 2*ZEROS; i++)
    {
      wm->padded_upsample_input[i] = 0;
    }

    for (i = 0; i < 2 * VALID_LENGTH + 2*ZEROS; i++)
    {
      wm->padded_upsample_double[i] = 0;
      wm->padded_half_input[i] = 0;
    }

    for (int i = 0; i < VALID_LENGTH * 4; i++)
    {
      wm->skip_1[i] = 0;
      wm->skip_2[i] = 0;
      wm->skip_3[i] = 0;
      wm->skip_4[i] = 0;
      wm->upsampled_input[i] = 0;
    }
        
    for (int i = 0; i < VALID_LENGTH * 8; i++)
    {
      wm->memory_grid[i] = 0;
      wm->memory_grid2[i] = 0;
    }
}


// Holds the malloc'd data
typedef struct {
    double* input;
    double* output;
    DenoiserState* ds;
    WorkingMemory* wm;
    LSTM_weights* lstmw;
    LSTM_Working_Memory* lstmwm;
} AllocatedMemory;

static AllocatedMemory am;

extern "C" void JNICALL
Java_com_example_hearingaidapplication_TestStreaming_00024Network_init(JNIEnv *env,
jobject thiz)
{
    am.input = (double*) malloc(INPUT_SIZE * sizeof(double));
    am.output = (double*) malloc(INPUT_SIZE * sizeof(double));
    for (int i=0; i < INPUT_SIZE; i++)
    {
        am.input[i] = 0;
        am.output[i] = 0;
    }

    am.ds = (DenoiserState*) malloc(sizeof(DenoiserState));
    am.wm = (WorkingMemory*) malloc(sizeof(WorkingMemory));
    am.lstmw = init_weights(0);
    am.lstmwm = init_LSTM_Working_Memory(0);
    mallocDenoiserState(am.ds);
    mallocWorkingMemory(am.wm);
    initializeWorkingMemory(am.wm);

    loadWeightsFromDisk(WEIGHTS_FILENAME, am.ds, am.lstmw);
}

extern "C" void JNICALL
Java_com_example_hearingaidapplication_TestStreaming_00024Network_freeMem(JNIEnv *env,
                                                                       jobject thiz)
{
    freeDenoiserState(am.ds);
    freeWorkingMemory(am.wm);
    free_LSTM_weights(am.lstmw);
    free_LSTM_Working_Memory(am.lstmwm);

    free(am.output);
}

/*extern "C" jdoubleArray JNICALL
Java_com_example_hearingaidapplication_TestStreaming_00024Network_test(JNIEnv *env,
                                                                              jobject thiz)
{
    FILE* file = fopen("weights.txt", "r");
    return file->
};*/



extern "C" jdoubleArray JNICALL
Java_com_example_hearingaidapplication_TestStreaming_00024Network_runDenoiser(JNIEnv *env,
                                                                              jobject thiz,
                                                                              jdoubleArray input) {

  // Run Denoiser

  // send input to am.input
  //env->GetDoubleField(input, 0);
  am.input = env->GetDoubleArrayElements(input, 0);

    //using namespace std::chrono;
    //milliseconds start = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    runDenoiser(am.input, am.ds, am.wm, am.output, am.lstmw, am.lstmwm);
    //milliseconds end = duration_cast<milliseconds>(system_clock::now().time_since_epoch());

    jdoubleArray out = env->NewDoubleArray(INPUT_SIZE);
    env->SetDoubleArrayRegion(out, 0, INPUT_SIZE, am.output);
    return out;

}