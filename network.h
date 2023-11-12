#include <math.h>
//#include <torch/torch.h>

// Number of zero padding in up/down sampling
const int ZEROS = 56;
// Size of one batch of input
const int INPUT_SIZE = 100000;
const int HALF_INPUT_SIZE = ceil(INPUT_SIZE / 2.0);

const int RESAMPLE = 4;

const bool NORMALIZE = true;

const double FLOOR = 0.001;

const int DEPTH = 4;
const int LAYERS = 2;
const int HIDDEN = 32;
const int KERNEL = 8;
const int STRIDE = 4;

// Constants for the network
double kernel_upsample[] = {-0.00000112810926111706,
0.00001033335865940899,
-0.00002920969745900948,
0.00005824976324220188,
-0.00009795445657800883,
0.00014883311814628541,
-0.00021140623721294105,
0.00028620564262382686,
-0.00037377749686129391,
0.00047468344564549625,
-0.00058950431412085891,
0.00071884220233187079,
-0.00086332455975934863,
0.00102360849268734455,
-0.00120038539171218872,
0.00139438582118600607,
-0.00160638894885778427,
0.00183722574729472399,
-0.00208779308013617992,
0.00235906033776700497,
-0.00265208492055535316,
0.00296802516095340252,
-0.00330815790221095085,
0.00367390131577849388,
-0.00406683608889579773,
0.00448874151334166527,
-0.00494162319228053093,
0.00542776286602020264,
-0.00594976916909217834,
0.00651064142584800720,
-0.00711385300382971764,
0.00776344956830143929,
-0.00846417434513568878,
0.00922162737697362900,
-0.01004246715456247330,
0.01093467604368925095,
-0.01190790720283985138,
0.01297392509877681732,
-0.01414723787456750870,
0.01544590946286916733,
-0.01689272373914718628,
0.01851681992411613464,
-0.02035603858530521393,
0.02246041968464851379,
-0.02489751577377319336,
0.02776070870459079742,
-0.03118285350501537323,
0.03535946458578109741,
-0.04059052094817161560,
0.04736081138253211975,
-0.05650795996189117432,
0.06961449235677719116,
-0.09007193148136138916,
0.12669886648654937744,
-0.21183113753795623779,
0.63649451732635498047,
0.63649451732635498047,
-0.21183112263679504395,
0.12669886648654937744,
-0.09007193148136138916,
0.06961449980735778809,
-0.05650795996189117432,
0.04736081138253211975,
-0.04059052467346191406,
0.03535946458578109741,
-0.03118285350501537323,
0.02776070684194564819,
-0.02489751391112804413,
0.02246041968464851379,
-0.02035603858530521393,
0.01851681992411613464,
-0.01689272373914718628,
0.01544590853154659271,
-0.01414723694324493408,
0.01297392416745424271,
-0.01190790720283985138,
0.01093467790633440018,
-0.01004246715456247330,
0.00922162737697362900,
-0.00846417527645826340,
0.00776345049962401390,
-0.00711385346949100494,
0.00651064142584800720,
-0.00594976870343089104,
0.00542776286602020264,
-0.00494162272661924362,
0.00448874104768037796,
-0.00406683608889579773,
0.00367390038445591927,
-0.00330815743654966354,
0.00296802422963082790,
-0.00265208445489406586,
0.00235905963927507401,
-0.00208779214881360531,
0.00183722504880279303,
-0.00160638801753520966,
0.00139438651967793703,
-0.00120038562454283237,
0.00102360872551798820,
-0.00086332479258999228,
0.00071884220233187079,
-0.00058950431412085891,
0.00047468344564549625,
-0.00037377749686129391,
0.00028620564262382686,
-0.00021140623721294105,
0.00014883311814628541,
-0.00009795426740311086,
0.00005824976324220188,
-0.00002920969745900948,
0.00001033335865940899,
-0.00000112810926111706};
double kernel_downsample[] = {-0.00000112810926111706,
0.00001033335865940899,
-0.00002920969745900948,
0.00005824976324220188,
-0.00009795445657800883,
0.00014883311814628541,
-0.00021140623721294105,
0.00028620564262382686,
-0.00037377749686129391,
0.00047468344564549625,
-0.00058950431412085891,
0.00071884220233187079,
-0.00086332455975934863,
0.00102360849268734455,
-0.00120038539171218872,
0.00139438582118600607,
-0.00160638894885778427,
0.00183722574729472399,
-0.00208779308013617992,
0.00235906033776700497,
-0.00265208492055535316,
0.00296802516095340252,
-0.00330815790221095085,
0.00367390131577849388,
-0.00406683608889579773,
0.00448874151334166527,
-0.00494162319228053093,
0.00542776286602020264,
-0.00594976916909217834,
0.00651064142584800720,
-0.00711385300382971764,
0.00776344956830143929,
-0.00846417434513568878,
0.00922162737697362900,
-0.01004246715456247330,
0.01093467604368925095,
-0.01190790720283985138,
0.01297392509877681732,
-0.01414723787456750870,
0.01544590946286916733,
-0.01689272373914718628,
0.01851681992411613464,
-0.02035603858530521393,
0.02246041968464851379,
-0.02489751577377319336,
0.02776070870459079742,
-0.03118285350501537323,
0.03535946458578109741,
-0.04059052094817161560,
0.04736081138253211975,
-0.05650795996189117432,
0.06961449235677719116,
-0.09007193148136138916,
0.12669886648654937744,
-0.21183113753795623779,
0.63649451732635498047,
0.63649451732635498047,
-0.21183112263679504395,
0.12669886648654937744,
-0.09007193148136138916,
0.06961449980735778809,
-0.05650795996189117432,
0.04736081138253211975,
-0.04059052467346191406,
0.03535946458578109741,
-0.03118285350501537323,
0.02776070684194564819,
-0.02489751391112804413,
0.02246041968464851379,
-0.02035603858530521393,
0.01851681992411613464,
-0.01689272373914718628,
0.01544590853154659271,
-0.01414723694324493408,
0.01297392416745424271,
-0.01190790720283985138,
0.01093467790633440018,
-0.01004246715456247330,
0.00922162737697362900,
-0.00846417527645826340,
0.00776345049962401390,
-0.00711385346949100494,
0.00651064142584800720,
-0.00594976870343089104,
0.00542776286602020264,
-0.00494162272661924362,
0.00448874104768037796,
-0.00406683608889579773,
0.00367390038445591927,
-0.00330815743654966354,
0.00296802422963082790,
-0.00265208445489406586,
0.00235905963927507401,
-0.00208779214881360531,
0.00183722504880279303,
-0.00160638801753520966,
0.00139438651967793703,
-0.00120038562454283237,
0.00102360872551798820,
-0.00086332479258999228,
0.00071884220233187079,
-0.00058950431412085891,
0.00047468344564549625,
-0.00037377749686129391,
0.00028620564262382686,
-0.00021140623721294105,
0.00014883311814628541,
-0.00009795426740311086,
0.00005824976324220188,
-0.00002920969745900948,
0.00001033335865940899,
-0.00000112810926111706};

struct Matrix
{
   double* data;
   int rows;
   int cols;
};

struct Matrix* matrix_init(int rows, int cols)
{
   struct Matrix* m = (struct Matrix*) malloc(sizeof(struct Matrix));
   m->rows = rows;
   m->cols = cols;
   m->data = (double*) malloc(rows * cols * sizeof(double));
   return m;
}

void matrix_free(struct Matrix* m)
{
   free(m->data);
   free(m);
}

void matrix_print(struct Matrix* m)
{
   int i, j;
   for (i = 0; i < m->rows; i++)
   {
      for (j = 0; j < m->cols; j++)
      {
         printf("%f ", m->data[i * m->cols + j]);
      }
      printf("\n");
   }
}

double inline matrix_get(struct Matrix* m, int row, int col)
{
   return m->data[row * m->cols + col];
}

void inline matrix_set(struct Matrix* m, int row, int col, double val)
{
   m->data[row * m->cols + col] = val;
}

int valid_length()
{
   int length = ceil(INPUT_SIZE * RESAMPLE);
   int idx;
   for (idx = 0; idx < DEPTH; idx++)
   {
      // Need to make sure we're not doing integer division.
      length = ceil((length - KERNEL) / (STRIDE * 1.0)) + 1;
      length = length > 1 ? length : 1;
   }
   for (idx = 0; idx < DEPTH; idx++)
   {
      length = (length - 1) * STRIDE + KERNEL;
   }
   length = ceil(length / RESAMPLE);
   return length;
}

const int VALID_LENGTH = valid_length();

// This struct contains arrays to be reused by functions.
struct WorkingMemory {
   // Should be of length VALID_LENGTH
   double* padded_input;
   double* upsampled_input;

   // These four arrays are used in upsampling.
   // Should be length VALID_LENGTH 
   double* upsample_working;
   // This should be length VALID_LENGTH + 112. 56 zeros on each side.
   double* padded_upsample_input;
   // Should be length 2*VALID_LENGTH
   double* upsample_working_double;
   // This should be length 2* VALID_LENGTH + 112. 56 zeros on each side.
   double* padded_upsample_double;

   double* half_input_one;
   double* half_input_two;
   double* padded_half_input;

   // Grid of numbers for use during en/decoding.
   double* memory_grid;
   double* memory_grid2;

   // Skips array.
   double* skip_1;
   double* skip_2;
   double* skip_3;
   double* skip_4;
};

void mallocWorkingMemory(WorkingMemory* wm)
{
   // Should be of length VALID_LENGTH
   wm->padded_input = (double*) malloc((VALID_LENGTH) * sizeof(double)); //
   wm->upsampled_input = (double*) malloc(VALID_LENGTH * 4 * sizeof(double)); //

   // These four arrays are used in upsampling.
   // Should be length VALID_LENGTH 
   wm->upsample_working = (double*) malloc((VALID_LENGTH) * sizeof(double)); //
   // This should be length VALID_LENGTH + 112. 56 zeros on each side.
   wm->padded_upsample_input = (double*) malloc((VALID_LENGTH + 2*ZEROS) * sizeof(double)); //
   // Should be length 2*VALID_LENGTH
   wm->upsample_working_double = (double*) malloc((2*VALID_LENGTH) * sizeof(double)); //
   // This should be length 2* VALID_LENGTH + 112. 56 zeros on each side.
   wm->padded_upsample_double = (double*) malloc((2*VALID_LENGTH + 2*ZEROS) * sizeof(double)); //

   wm->half_input_one = (double*) malloc(VALID_LENGTH * 2 * sizeof(double));
   wm->half_input_two = (double*) malloc(VALID_LENGTH * 2 * sizeof(double));
   wm->padded_half_input = (double*) malloc((VALID_LENGTH * 2 + 2*ZEROS) * sizeof(double));

   // Grid of numbers for use during en/decoding.
   wm->memory_grid = (double*) malloc(VALID_LENGTH * 8 * sizeof(double));
   wm->memory_grid2 = (double*) malloc(VALID_LENGTH * 8 * sizeof(double));

   // Skips array.
   wm->skip_1 = (double*) malloc(4 * VALID_LENGTH * sizeof(double));
   wm->skip_2 = (double*) malloc(4 * VALID_LENGTH * sizeof(double));
   wm->skip_3 = (double*) malloc(4 * VALID_LENGTH * sizeof(double));
   wm->skip_4 = (double*) malloc(4 * VALID_LENGTH * sizeof(double));

}

void freeWorkingMemory(WorkingMemory* wm)
{
  free(wm->padded_input);
  free(wm->upsampled_input);
  
   free(wm->upsample_working);
   free(wm->padded_upsample_input);
   free(wm->upsample_working_double);
   free(wm->padded_upsample_double);

   free(wm->half_input_one);
   free(wm->half_input_two);
   free(wm->padded_half_input);

   free(wm->memory_grid);
   free(wm->memory_grid2);

   free(wm->skip_1);
   free(wm->skip_2);
   free(wm->skip_3);
   free(wm->skip_4);

   free(wm);
}

// These will all only be written to when weights are loaded.
struct DenoiserState
{

   // 4 * 1 * 8 2D array.
   double* encoder_0_0_weight;
   // 4 long bias array.
   double* encoder_0_0_bias;
   // 8 * 4 * 1 3D array.
   double* encoder_0_2_weight;
   // 8 long bias array.
   double* encoder_0_2_bias;

   // 8 * 4 * 8 3D array
   double* encoder_1_0_weight;
   // 8 long bias array.
   double* encoder_1_0_bias;
   // 16 * 8 * 1 3D array
   double* encoder_1_2_weight;
   // 16 long bias array.
   double* encoder_1_2_bias;

   // 16 * 8 * 8 3D array
   double* encoder_2_0_weight;
   // 16 long bias array.
   double* encoder_2_0_bias;
   // 32 * 16 * 1 3D array
   double* encoder_2_2_weight;
   // 32 long bias array.
   double* encoder_2_2_bias;

   // 32 * 16 * 8 3D array
   double* encoder_3_0_weight;
   // 32 long bias array.
   double* encoder_3_0_bias;
   // 32 * 16 * 1 3D array
   double* encoder_3_2_weight;
   // 32 long bias array.
   double* encoder_3_2_bias;

   // 64 * 32 * 1 3D array
   double* decoder_0_0_weight;
   // 64 long bias array.
   double* decoder_0_0_bias;
   // 32 * 16 * 8 3D array 
   double* decoder_0_2_weight;
   // 32 long bias array.
   double* decoder_0_2_bias;

   // 32 * 16 * 1 3D array
   double* decoder_1_0_weight;
   // 32 long bias array.
   double* decoder_1_0_bias;
   // 16 * 8 * 8 3D array 
   double* decoder_1_2_weight;
   // 8 long bias array.
   double* decoder_1_2_bias;

   // 16 * 8 * 1 3D array
   double* decoder_2_0_weight;
   // 16 long bias array.
   double* decoder_2_0_bias;
   // 8 * 4 * 8 3D array 
   double* decoder_2_2_weight;
   // 4 long bias array.
   double* decoder_2_2_bias;

   // 8 * 4 * 1 3D array
   double* decoder_3_0_weight;
   // 8 long bias array.
   double* decoder_3_0_bias;
   // 8 * 4 * 8 3D array 
   double* decoder_3_2_weight;
   // 4 long bias array.
   double* decoder_3_2_bias;
};

void mallocDenoiserState(DenoiserState* ds)
{
   ds->encoder_0_0_weight = (double*) malloc((4 * 8) * sizeof(double));
   ds->encoder_0_0_bias = (double*) malloc((4) * sizeof(double));
   ds->encoder_0_2_weight = (double*) malloc((8 * 4 * 1) * sizeof(double));
   ds->encoder_0_2_bias = (double*) malloc((8) * sizeof(double));

   ds->encoder_1_0_weight = (double*) malloc((8 * 4 * 8) * sizeof(double));
   ds->encoder_1_0_bias = (double*) malloc((8) * sizeof(double));
   ds->encoder_1_2_weight = (double*) malloc((16 * 8 * 1) * sizeof(double));
   ds->encoder_1_2_bias = (double*) malloc((16) * sizeof(double));

   ds->encoder_2_0_weight = (double*) malloc((16 * 8 * 8) * sizeof(double));
   ds->encoder_2_0_bias = (double*) malloc((16) * sizeof(double));
   ds->encoder_2_2_weight = (double*) malloc((32 * 16 * 1) * sizeof(double));
   ds->encoder_2_2_bias = (double*) malloc((32) * sizeof(double));

   ds->encoder_3_0_weight = (double*) malloc((32 * 16 * 8) * sizeof(double));
   ds->encoder_3_0_bias = (double*) malloc((32) * sizeof(double));
   ds->encoder_3_2_weight = (double*) malloc((64 * 32 * 1) * sizeof(double));
   ds->encoder_3_2_bias = (double*) malloc((64) * sizeof(double));

   ds->decoder_0_0_weight = (double*) malloc((64 * 32 * 1) * sizeof(double));
   ds->decoder_0_0_bias = (double*) malloc((64) * sizeof(double));
   ds->decoder_0_2_weight = (double*) malloc((32 * 16 * 8) * sizeof(double));
   ds->decoder_0_2_bias = (double*) malloc((16) * sizeof(double));

   ds->decoder_1_0_weight = (double*) malloc((32 * 16 * 1) * sizeof(double));
   ds->decoder_1_0_bias = (double*) malloc((32) * sizeof(double));
   ds->decoder_1_2_weight = (double*) malloc((16 * 8 * 8) * sizeof(double));
   ds->decoder_1_2_bias = (double*) malloc((8) * sizeof(double));

   ds->decoder_2_0_weight = (double*) malloc((16 * 8 * 1) * sizeof(double));
   ds->decoder_2_0_bias = (double*) malloc((16) * sizeof(double));
   ds->decoder_2_2_weight = (double*) malloc((8 * 4 * 8) * sizeof(double));
   ds->decoder_2_2_bias = (double*) malloc((4) * sizeof(double));

   ds->decoder_3_0_weight = (double*) malloc((8 * 4 * 1) * sizeof(double));
   ds->decoder_3_0_bias = (double*) malloc((8) * sizeof(double));
   ds->decoder_3_2_weight = (double*) malloc((4 * 1 * 8) * sizeof(double));
   ds->decoder_3_2_bias = (double*) malloc((1) * sizeof(double));
}

// Destroys the denoiser state struct.
void freeDenoiserState(DenoiserState* ds)
{

   free(ds->decoder_3_0_weight);
   free(ds->decoder_3_0_bias);
   free(ds->decoder_3_2_weight);
   free(ds->decoder_3_2_bias);

   free(ds->decoder_2_0_weight);
   free(ds->decoder_2_0_bias);
   free(ds->decoder_2_2_weight);
   free(ds->decoder_2_2_bias);

   free(ds->decoder_1_0_weight);
   free(ds->decoder_1_0_bias);
   free(ds->decoder_1_2_weight);
   free(ds->decoder_1_2_bias);

   free(ds->decoder_0_0_weight);
   free(ds->decoder_0_0_bias);
   free(ds->decoder_0_2_weight);
   free(ds->decoder_0_2_bias);

   free(ds->encoder_3_0_weight);
   free(ds->encoder_3_0_bias); 
   free(ds->encoder_3_2_weight);
   free(ds->encoder_3_2_bias);

   free(ds->encoder_2_0_weight);
   free(ds->encoder_2_0_bias);
   free(ds->encoder_2_2_weight);
   free(ds->encoder_2_2_bias);

   free(ds->encoder_1_0_weight);
   free(ds->encoder_1_0_bias);
   free(ds->encoder_1_2_weight);
   free(ds->encoder_1_2_bias);

   free(ds->encoder_0_0_weight);
    free(ds->encoder_0_0_bias);
   free(ds->encoder_0_2_weight);
    free(ds->encoder_0_2_bias);

   free(ds);
}