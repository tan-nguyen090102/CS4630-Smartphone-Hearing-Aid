#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "network.h"
#include <iostream>
#include <vector>
#include <sndfile.h>
#include "LSTM.h"

//#include <torch/torch.h>

void print_array(double* arr, int start, int end, std::string name) {
  std::cout << name << std::endl;
  for (int i=start; i < end; i++) {
    printf("%i:%f\n", i, arr[i]);
  }
}

// Loads a .wav file into an array of doubles.
SF_INFO loadWavFile(char* filename, std::vector<double> &audioData) {
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
}

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

void runDenoiser(double* inp, DenoiserState* ds, WorkingMemory* wm, double* output)
{
  // Run denoiser
  printf("Normalizing...\n");
  double SD; // Standard deviation
  if (NORMALIZE)
  {
    double sum = 0;
    int i;
    for (i=0; i < INPUT_SIZE; i++) {
        sum += inp[i];
    }
    std::cout << "Sum: " << sum << std::endl;
    sum /= INPUT_SIZE; // This is the average.
    SD = 0;

    for (i = 0; i < INPUT_SIZE; ++i) {
        SD += pow(inp[i] - sum, 2);
    }
    // For some reason, pytorch uses N-1 instead of N.
    SD = sqrt(SD / (INPUT_SIZE - 1));

    std::cout << "SD: " << SD << std::endl;

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
  printf("Upsampling data...\n");
  double_upsample2_valid(wm->padded_input, wm->upsampled_input, wm);

  // Now we run each of the encoders in sequence.

  // encoder.0.0
  printf("Encoder.0.0\n");

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
  printf("Encoder.1.0\n");
  
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
  printf("Encoder.2.0\n");
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
  printf("Encoder.3.0\n");
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

  printf("Run LSTM\n");
  // Currently not implemented. Need to implement this.
  

  // Add skip4
  for (int i=0; i < current_length * 32; i++) {
    wm->memory_grid2[i] += wm->skip_4[i];
  }
  
  // decoder.0.0
  printf("Decoder.0.0\n");
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
  printf("Decoder.1.0\n");
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
  printf("Decoder.2.0\n");
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
  printf("Decoder.3.0\n");
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
void loadWeightsFromDisk(char* filename, DenoiserState* ds)
{
  std::cout << "Loading weight file" << std::endl;
  FILE* file = fopen(filename, "r");

  for (int i = 0; i < 4 * 8; i++)
  {
    fscanf(file, "%lf", &ds->encoder_0_0_weight[i]);
  }
    
  for (int i = 0; i < 4; i++)
    fscanf(file, "%lf", &ds->encoder_0_0_bias[i]);

  for (int i = 0; i < 4 * 8; i++)
    fscanf(file, "%lf", &ds->encoder_0_2_weight[i]);
  for (int i = 0; i < 8; i++)
    fscanf(file, "%lf", &ds->encoder_0_2_bias[i]);


  for (int i = 0; i < 8 * 4 * 8; i++)
    fscanf(file, "%lf", &ds->encoder_1_0_weight[i]);
  for (int i = 0; i < 8; i++)
    fscanf(file, "%lf", &ds->encoder_1_0_bias[i]);

  for (int i = 0; i < 16 * 8; i++)
    fscanf(file, "%lf", &ds->encoder_1_2_weight[i]);
  for (int i = 0; i < 16; i++)
    fscanf(file, "%lf", &ds->encoder_1_2_bias[i]);


  for (int i = 0; i < 16 * 8 * 8; i++)
    fscanf(file, "%lf", &ds->encoder_2_0_weight[i]);
  for (int i = 0; i < 16; i++)
    fscanf(file, "%lf", &ds->encoder_2_0_bias[i]);

  for (int i = 0; i < 32 * 16; i++)
    fscanf(file, "%lf", &ds->encoder_2_2_weight[i]);
  for (int i = 0; i < 32; i++)
    fscanf(file, "%lf", &ds->encoder_2_2_bias[i]);


  for (int i = 0; i < 32 * 16 * 8; i++)
    fscanf(file, "%lf", &ds->encoder_3_0_weight[i]);
  for (int i = 0; i < 32; i++)
    fscanf(file, "%lf", &ds->encoder_3_0_bias[i]);

  for (int i = 0; i < 64 * 32; i++)
    fscanf(file, "%lf", &ds->encoder_3_2_weight[i]);
  for (int i = 0; i < 64; i++)
    fscanf(file, "%lf", &ds->encoder_3_2_bias[i]);

  
  for (int i = 0; i < 64 * 32 * 1; i++)
    fscanf(file, "%lf", &ds->decoder_0_0_weight[i]);
  for (int i = 0; i < 64; i++)
    fscanf(file, "%lf", &ds->decoder_0_0_bias[i]);

  for (int i = 0; i < 32 * 16 * 8; i++)
    fscanf(file, "%lf", &ds->decoder_0_2_weight[i]);
  for (int i = 0; i < 16; i++)
    fscanf(file, "%lf", &ds->decoder_0_2_bias[i]);


  for (int i = 0; i < 32 * 16 * 1; i++)
    fscanf(file, "%lf", &ds->decoder_1_0_weight[i]);
  for (int i = 0; i < 32; i++)
    fscanf(file, "%lf", &ds->decoder_1_0_bias[i]);

  for (int i = 0; i < 16 * 8 * 8; i++)
    fscanf(file, "%lf", &ds->decoder_1_2_weight[i]);
  for (int i = 0; i < 8; i++)
    fscanf(file, "%lf", &ds->decoder_1_2_bias[i]);


  for (int i = 0; i < 16 * 8 * 1; i++)
    fscanf(file, "%lf", &ds->decoder_2_0_weight[i]);
  for (int i = 0; i < 16; i++)
    fscanf(file, "%lf", &ds->decoder_2_0_bias[i]);

  for (int i = 0; i < 8 * 4 * 8; i++)
    fscanf(file, "%lf", &ds->decoder_2_2_weight[i]);
  for (int i = 0; i < 4; i++)
    fscanf(file, "%lf", &ds->decoder_2_2_bias[i]);

  
  for (int i = 0; i < 8 * 4 * 1; i++)
    fscanf(file, "%lf", &ds->decoder_3_0_weight[i]);
  for (int i = 0; i < 8; i++)
    fscanf(file, "%lf", &ds->decoder_3_0_bias[i]);

  for (int i = 0; i < 4 * 1 * 8; i++)
    fscanf(file, "%lf", &ds->decoder_3_2_weight[i]);
  for (int i = 0; i < 1; i++)
    fscanf(file, "%lf", &ds->decoder_3_2_bias[i]);

  fclose(file);
  
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


int main(int argc, char *argv[]) {

  int operation = 5; // 0 = denoiser, 1 = convolution tests, 2/3 = speed/sparsity tests, 4 = load wav file

  if (argc > 1)
  {
    operation = atoi(argv[1]);
  }
  else
  {
    printf("No operation specified. Defaulting to denoiser.\n");
    operation = 0;
  }

  // Run Denoiser
  if (operation == 0)
  {
    double* input = (double*) malloc(INPUT_SIZE * sizeof(double));
    double* output = (double*) malloc(INPUT_SIZE * sizeof(double));
    for (int i=0; i < INPUT_SIZE; i++) 
    {
        input[i] = 0;
        output[i] = 0;
    }

     SF_INFO sfinfo;

    if (argc > 2)
    {
      std::cout << "Loading wav file" << std::endl;
      std::vector<double> audioData;
      sfinfo = loadWavFile(argv[2], audioData);
      vectorToArray(audioData, input, INPUT_SIZE);
      std::cout << "Loaded wav file" << std::endl;
    }
    else
    {
      std::cout << "No wav file specified. Defaulting to sine wave." << std::endl;
      for (int i=0; i < INPUT_SIZE; i++) 
      {
          input[i] = sin(i * .1);
      }
    }

    DenoiserState *ds = (DenoiserState*) malloc(sizeof(DenoiserState));
    WorkingMemory *wm = (WorkingMemory*) malloc(sizeof(WorkingMemory));
    //DenoiserStatePyTorch dspt = DenoiserStatePyTorch();
    mallocDenoiserState(ds);
    mallocWorkingMemory(wm);

    std::cout << "Allocated memory" << std::endl;

    if (argc > 3)
    {
      std::cout << "Loading weights from disk" << std::endl;
      loadWeightsFromDisk(argv[3], ds);
    }
    else
    {
      std::cout << "No weight file specified. Defaulting to random weights." << std::endl;
      initializeDenoiserState(ds, 0, 0.1);
      //initalizeDenoiserState(&dspt);
      randomizeWeights(0, ds);
    }

    initializeWorkingMemory(wm);

    std::cout << "Starting" << std::endl;
    
    runDenoiser(input, ds, wm, output);

    std::cout << "End" << std::endl;
    
    freeDenoiserState(ds);
    freeWorkingMemory(wm);

    // Save output as .wav file.
    // Add a readme.md

    if (argc > 4 )
    {
      std::cout << "Saving file to " << argv[4] << std::endl;

      SNDFILE * outfile = sf_open(argv[4], SFM_WRITE, &sfinfo);
      sf_count_t count = sf_write_double(outfile, &output[0], INPUT_SIZE);
      sf_write_sync(outfile);
      sf_close(outfile);
    }

    free(input);
    free(output);
    std::cout << "Done" << std::endl;
    return 0;
  }
  else if (operation == 1) // Run convolution tests
  {

    std::cout << "Starting convolution tests" << std::endl;
    const int KERNEL_SIZE = 64;
    const int OUTPUT_LENGTH = (INPUT_SIZE - KERNEL_SIZE) / STRIDE + 1;
    double* input = (double*) malloc(INPUT_SIZE * sizeof(double));
    double* kernel = (double*) malloc(KERNEL_SIZE * sizeof(double));
    double* output = (double*) malloc(OUTPUT_LENGTH * sizeof(double));

    std::cout << "Allocated memory" << std::endl;
    for (int i=0; i < INPUT_SIZE; i++) {
        input[i] = sin(i * .1);
    }

    for (int i=0; i < KERNEL_SIZE; i++) {
        kernel[i] = sin(i * -.1);
    }

    //conv1d_unrolled(input, kernel, 0, output, INPUT_SIZE, OUTPUT_LENGTH, KERNEL_SIZE, STRIDE);

    for (int i=0; i < OUTPUT_LENGTH; i++) {
        printf("%f\n", output[i]);
    }

    free(input);
    free(kernel);
    free(output);
  }
  else if (operation == 2)
  {

    /* Read the four main parameters of the LSTM network  */

    std::cout << "1" << std::endl;

    double parameters[4];
    parameters[0] = 1;
    parameters[1] = 1;
    parameters[2] = 2;
    parameters[3] = 32;
    int timesteps = 10;
    int no_of_batches = 1;
    int no_of_layers = 2;
    int no_of_units = 32;

	/* Read the weights of LSTM network and save them in a structure NN_model  */

    struct LSTM_parameters NN_model;
    char c1[50] = "weights.txt";

    std::cout << "2" << std::endl;
	  read_matrices(c1, &NN_model, no_of_units);
    std::cout << "3" << std::endl;

	/* Define the rest of the variables */

    int count = 0; // counter to terminate computation
	  double result; //intermidiate result from the neural network, our y value
    double observation; //single observation added to the batch
    int input_length = 100; // number of all the values passed in input.txt file
    double *sequence = (double *)calloc(timesteps, timesteps * sizeof(double)); // dummy variable used for implementation
    double *single_batch = (double *)calloc(timesteps, timesteps * sizeof(double)); // one batch of observations

    double *input_sequence = (double *)calloc(input_length, input_length * sizeof(double));
    double *total_output = (double *)calloc(input_length, input_length * sizeof(double));

    std::cout << "4" << std::endl;
    
    for (int i=0; i < input_length; i++) {
        input_sequence[i] = sin(i * .1);
    }

    std::cout << "5" << std::endl;

    while (count < input_length )
	  { // iterate over batches

        observation = input_sequence[count];
        std::cout << "6" << std::endl;
        input_value_to_vector(observation, timesteps, single_batch, sequence); // append a single observation to the batch
        std::cout << "7" << std::endl;
        result = LSTM_algorithm_three_LSTM_layers (single_batch, parameters, NN_model); // run the implementation
        std::cout << "8" << std::endl;
        printf("The intermidiate result is:     %f\n", result);
        total_output[count] = result;
        count = count+1;
        
	  }

    print_array(total_output, 0, 100, "Total output");

    return 0;
  }
  return 0;
}


