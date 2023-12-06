#include <math.h>
#include <string>
#include <iostream>

// Number of zero padding in up/down sampling
const int ZEROS = 56;
// Size of one batch of input
const int INPUT_SIZE = 160000;
const int HALF_INPUT_SIZE = ceil(INPUT_SIZE / 2.0);

const int RESAMPLE = 4;

const bool NORMALIZE = true;

const double FLOOR = 0.001;

const int KERNEL = 8;
const int STRIDE = 4;

const int LSTM_HIDDEN_SIZE = 32;
const int LSTM_NUM_LAYERS = 2;
const int LSTM_BATCH_SIZE = 1;
const int DEPTH = 4;


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

const int LSTM_INPUT_SIZE = 32;
const int LSTM_NUM_SAMPLES = (((((VALID_LENGTH - 1) / 4) - 1) / 4) - 1) / 4 - 1;


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
    for (int i = 0; i < arr->y; i++) {
        for (int j = 0; j < arr->x; j++) {
            contents += std::to_string(arr->data[i * arr->y + j]) + " ";
        }
        contents += "\n";
    }
    return contents;
}

// Function to return array contents as a string
std::string getVectorContents(struct Vector* arr) {
    std::string contents = "";
    for (int i = 0; i < arr->size; i++) {
        contents += std::to_string(arr->data[i]) + " ";
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

// Function to copy one array to another
void copyArray2D(struct Array2D* x, struct Array2D* y) {
    for (int i = 0; i < x->x; i++) {
        for (int j = 0; j < x->y; j++) {
            set(y, i, j, get(x, i, j));
        }
    }
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

// X and Y are LSTM_HIDDEN_SIZE by LSTM_BATCH_SIZE
// Z is LSTM_HIDDEN_SIZE long.
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
    Array2D** weight_ih; // Should be LSTM_NUM_LAYERS by 4 * LSTM_HIDDEN_SIZE by LSTM_INPUT_SIZE
    Array2D** weight_hh; // Should be LSTM_NUM_LAYERS by 4 * LSTM_HIDDEN_SIZE by LSTM_HIDDEN_SIZE

    Vector** ibias_input; // Should be LSTM_NUM_LAYERS by LSTM_HIDDEN_SIZE. This is b_ii
    Vector** ibias_forget; // Should be LSTM_NUM_LAYERS by LSTM_HIDDEN_SIZE. This is b_if
    Vector** ibias_gate; // Should be LSTM_NUM_LAYERS by LSTM_HIDDEN_SIZE. This is b_ig
    Vector** ibias_output; // Should be LSTM_NUM_LAYERS by LSTM_HIDDEN_SIZE. This is b_io

    Vector** hbias_input; // Should be LSTM_NUM_LAYERS by LSTM_HIDDEN_SIZE. This is b_hi
    Vector** hbias_forget; // Should be LSTM_NUM_LAYERS by LSTM_HIDDEN_SIZE. This is b_hf
    Vector** hbias_gate; // Should be LSTM_NUM_LAYERS by LSTM_HIDDEN_SIZE. This is b_hg
    Vector** hbias_output; // Should be LSTM_NUM_LAYERS by LSTM_HIDDEN_SIZE. This is b_ho

};

// Mallocs all the arrays for the weights and biases
LSTM_weights* init_weights(double init)
{
    LSTM_weights* weights = (LSTM_weights*) malloc(sizeof(struct LSTM_weights));

    // Allocate a load of arrays of pointers to 2D arrays

    weights->weight_ih = (Array2D**) malloc(LSTM_NUM_LAYERS * sizeof(struct Array2D*));
    weights->weight_hh = (Array2D**) malloc(LSTM_NUM_LAYERS * sizeof(struct Array2D*));

    weights->ibias_input = (Vector**) malloc(LSTM_NUM_LAYERS * sizeof(struct Vector*));
    weights->ibias_forget = (Vector**) malloc(LSTM_NUM_LAYERS * sizeof(struct Vector*));
    weights->ibias_gate = (Vector**) malloc(LSTM_NUM_LAYERS * sizeof(struct Vector*));
    weights->ibias_output = (Vector**) malloc(LSTM_NUM_LAYERS * sizeof(struct Vector*));

    weights->hbias_input = (Vector**) malloc(LSTM_NUM_LAYERS * sizeof(struct Vector*));
    weights->hbias_forget = (Vector**) malloc(LSTM_NUM_LAYERS * sizeof(struct Vector*));
    weights->hbias_gate = (Vector**) malloc(LSTM_NUM_LAYERS * sizeof(struct Vector*));
    weights->hbias_output = (Vector**) malloc(LSTM_NUM_LAYERS * sizeof(struct Vector*));

    for (int i = 0; i < LSTM_NUM_LAYERS; i++)
    {
        if (i == 0)
        {
            weights->weight_ih[i] = createArray2D( 4 * LSTM_HIDDEN_SIZE, LSTM_INPUT_SIZE);
            
        }
        else
        {
            weights->weight_ih[i] = createArray2D(4 * LSTM_HIDDEN_SIZE,LSTM_HIDDEN_SIZE);
        }
        fillArray2D(weights->weight_ih[i], init);

        weights->weight_hh[i] = createArray2D(4 * LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE);
        fillArray2D(weights->weight_hh[i], init);

        weights->ibias_input[i] = createVector(LSTM_HIDDEN_SIZE);
        fillVector(weights->ibias_input[i], init);

        weights->ibias_forget[i] = createVector(LSTM_HIDDEN_SIZE);
        fillVector(weights->ibias_forget[i], init);

        weights->ibias_gate[i] = createVector(LSTM_HIDDEN_SIZE);
        fillVector(weights->ibias_gate[i], init);

        weights->ibias_output[i] = createVector(LSTM_HIDDEN_SIZE);
        fillVector(weights->ibias_output[i], init);


        weights->hbias_input[i] = createVector(LSTM_HIDDEN_SIZE);
        fillVector(weights->hbias_input[i], init);

        weights->hbias_forget[i] = createVector(LSTM_HIDDEN_SIZE);
        fillVector(weights->hbias_forget[i], init);

        weights->hbias_gate[i] = createVector(LSTM_HIDDEN_SIZE);
        fillVector(weights->hbias_gate[i], init);

        weights->hbias_output[i] = createVector(LSTM_HIDDEN_SIZE);
        fillVector(weights->hbias_output[i], init);

        
    }
    return weights;
}

void free_LSTM_weights(LSTM_weights* weights)
{
    for (int i = 0; i < LSTM_NUM_LAYERS; i++)
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
    hidden_state = createArray2D(LSTM_NUM_LAYERS, LSTM_HIDDEN_SIZE);
    cell_state = createArray2D(LSTM_NUM_LAYERS, LSTM_HIDDEN_SIZE);

    for (int i = 0; i < LSTM_NUM_LAYERS; i++)
    {
        for (int j = 0; j < LSTM_HIDDEN_SIZE; j++)
        {
            set(hidden_state, i, j, 0.0);
            set(cell_state, i, j, 0.0);
        }
    }
}

struct LSTM_Working_Memory
{
   Array2D** input_sequence;
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

    Array2D** cell_states;
    Array2D** hidden_states;
    Array2D** output_values;
};

LSTM_Working_Memory* init_LSTM_Working_Memory(double init)
{
    LSTM_Working_Memory* wm = (LSTM_Working_Memory*) malloc(sizeof(struct LSTM_Working_Memory));
    wm->input_sequence = (Array2D**) malloc(LSTM_NUM_SAMPLES * sizeof(struct Array2D*));
    wm->input_gate = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);
    wm->forget_gate = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);
    wm->output_gate = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);
    wm->gate_gate = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);

    wm->w_times_input = createArray2D(LSTM_HIDDEN_SIZE * 4, LSTM_BATCH_SIZE);
    wm->inp_slice = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);
    wm->forget_slice = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);
    wm->output_slice = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);
    wm->gate_slice = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);

    wm->h_times_state = createArray2D(LSTM_HIDDEN_SIZE * 4, LSTM_BATCH_SIZE);
    wm->h_inp_slice = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);
    wm->h_forget_slice = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);
    wm->h_output_slice = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);
    wm->h_gate_slice = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);

    wm->forget_times_cell = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);
    wm->input_times_gate = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);
    wm->tanh_cell = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);

    // Initialize all the arrays to 0
    for (int i = 0; i < LSTM_HIDDEN_SIZE; i++)
    {
        for (int j = 0; j < LSTM_BATCH_SIZE; j++)
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



    for (int i = 0; i < LSTM_HIDDEN_SIZE * 4; i++)
    {
        for (int j = 0; j < LSTM_BATCH_SIZE; j++)
        {
            set(wm->w_times_input, i, j, init);
            set(wm->h_times_state, i, j, init);
        }
    }

    wm->cell_states = (Array2D**) malloc(LSTM_NUM_LAYERS * sizeof(struct Array2D*));

    wm->hidden_states = (Array2D**) malloc(LSTM_NUM_LAYERS * sizeof(struct Array2D*));
    wm->output_values = (Array2D**) malloc(LSTM_NUM_SAMPLES * sizeof(struct Array2D*));

    
    for (int i = 0; i < LSTM_NUM_SAMPLES; i++)
    {
        wm->input_sequence[i] = createArray2D(LSTM_INPUT_SIZE, LSTM_BATCH_SIZE);
        wm->output_values[i] = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);
        fillArray2D(wm->output_values[i], 0);
    }
    for (int i = 0; i < LSTM_NUM_LAYERS; i++)
    {
        wm->hidden_states[i] = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);
        wm->cell_states[i] = createArray2D(LSTM_HIDDEN_SIZE, LSTM_BATCH_SIZE);
        fillArray2D(wm->hidden_states[i], 0);
        fillArray2D(wm->cell_states[i], 0);
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

    for (int i = 0; i < LSTM_NUM_SAMPLES; i++)
    {
        freeArray2D(wm->output_values[i]);
        freeArray2D(wm->input_sequence[i]);
    }
    free(wm->input_sequence);
    free(wm->output_values);

    for (int i = 0; i < LSTM_NUM_LAYERS; i++)
    {
        freeArray2D(wm->hidden_states[i]);
        freeArray2D(wm->cell_states[i]);
    }
    free(wm->hidden_states);
    free(wm->cell_states);

    free(wm);
}