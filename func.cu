//****************************************************************************
// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//****************************************************************************

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

#define GAUSSIAN_SZ 9
// 1 = Laplacian5x5 ; 2 = Nitidez5x5; 3 = PasoAlto5x5; 4 = Media3x3 ; 5 = Blur3x3 ; 6 = Blur5x5 ; 7 = GaussianBlur ; 8 = SobelHori3x3 ; 9 = SobelVert3x3
#define FILTER 7
//Definimos tamaño de bloque en preprocesador para facilidad al hacer pruebas
#define BLOCK_SZ 32
//Definimos tamaño de convolución en preprocesador para poder inicializar array de memoria constante
#if FILTER == 4 || FILTER == 5 || FILTER == 8 || FILTER == 9
#define KERNEL_SZ 3
#elif FILTER == 7
#ifndef GAUSSIAN_SZ
#define KERNEL_SZ 3
#else
#define KERNEL_SZ GAUSSIAN_SZ
#endif
#else
#define KERNEL_SZ 5
#endif
__constant__ float d_filterConst[KERNEL_SZ*KERNEL_SZ];
//Definimos para facilitar el cambio entre los kernels de memoria compartida y global
#define SHARED 1

__global__
void box_filter_shared(const unsigned char* const inputChannel,
  unsigned char* const outputChannel,
  int numRows, int numCols,
  const float* const filter, const int filterWidth)
{
  // TODO: 
  // NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la imagen
  //
  // if ( absolute_image_position_x >= numCols ||
  //      absolute_image_position_y >= numRows )
  // {
  //     return;
  // }
  // NOTA: Que un thread tenga una posición correcta en 2D no quiere decir que al aplicar el filtro
  // los valores de sus vecinos sean correctos, ya que pueden salirse de la imagen.

  extern __shared__ unsigned char image_shared[];

  const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y);
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
  //Shared size siempre debería ser par, ya que blockdim.x, blockdim.y
  //y filterwidth-1 siempre deberán ser pares
  const int sharedSize = (blockDim.x + filterWidth - 1) * (blockDim.y + filterWidth - 1);
  const int halfFilterWidth = filterWidth / 2;
  const int threadNum = threadIdx.y * blockDim.x + threadIdx.x;
  const int width = blockDim.x + filterWidth - 1;

  const int numThreads = blockDim.x * blockDim.y;
  int workingThreads, offset = 0;

  //Calculamos coordenadas de imagen de la sección a mapear en shared memory
  const int startX = blockIdx.x * blockDim.x - halfFilterWidth;
  const int startY = blockIdx.y * blockDim.y - halfFilterWidth;

  while(offset < sharedSize)
  {
    workingThreads = sharedSize - offset;
    workingThreads = numThreads > workingThreads ? workingThreads : numThreads;

    if(threadNum < workingThreads)
    {
      //Calculamos las coordenadas en shared memory
      int sharedY = (threadNum+offset) / width;
      int sharedX = (threadNum+offset) - sharedY * width;
      //Pasamos a coordenadas de imagen
      int imgX = sharedX + startX;
      int imgY = sharedY + startY;
      //Hacemos clamp para asegurar que no nos salimos de la imagen
      imgY = imgY >= numRows ? numRows - 1 : imgY < 0 ? 0 : imgY;
      imgX = imgX >= numCols ? numCols - 1 : imgX < 0 ? 0 : imgX;

      image_shared[threadNum + offset] = inputChannel[imgY * numCols + imgX];
    }
    offset += workingThreads;
  }

  __syncthreads();

  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;
  int filterRadius = filterWidth / 2;
  float result = 0;
  for (int j = -filterRadius; j <= filterRadius; j++)
    for (int i = -filterRadius; i <= filterRadius; i++) {
      int x = threadIdx.x + halfFilterWidth + i;
      int y = threadIdx.y + halfFilterWidth + j;

      result += (float)d_filterConst[(j + filterRadius)*filterWidth + i + filterRadius] * (float)image_shared[y*width + x];
    }
  outputChannel[thread_1D_pos] = result > 255 ? 255 : result < 0 ? 0 : (char)result;
}

__global__
void box_filter(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  // TODO: 
  // NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la imagen
  //
  // if ( absolute_image_position_x >= numCols ||
  //      absolute_image_position_y >= numRows )
  // {
  //     return;
  // }
  // NOTA: Que un thread tenga una posición correcta en 2D no quiere decir que al aplicar el filtro
  // los valores de sus vecinos sean correctos, ya que pueden salirse de la imagen.

  const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y);
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;
  int filterRadius = filterWidth / 2;
  float result = 0;
  for (int j = -filterRadius; j <= filterRadius; j++)
    for (int i = -filterRadius; i <= filterRadius; i++)  {
      int x = thread_2D_pos.x + i;
      x = x >= numCols ? numCols - 1 : x;
      x = x < 0 ? 0 : x;
      int y = thread_2D_pos.y + j;
      y = y >= numRows ? numRows - 1 : y;
      y = y < 0 ? 0 : y;
      //Sin memoria de constantes
      //result += (float) filter[(j + filterRadius)*filterWidth + i + filterRadius] * (float) inputChannel[y*numCols + x];
      //Con memoria de constantes
      result += (float)d_filterConst[(j + filterRadius)*filterWidth + i + filterRadius] * (float)inputChannel[y*numCols + x];
    }
  outputChannel[thread_2D_pos.y * numCols + thread_2D_pos.x] = result>255?255:result<0?0:(char)result;
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  // TODO: 
  // NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la imagen
  //
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
  greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
  blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
}

//This kernel takes in three color channels and recombines them
//into one image. The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //TODO:
  //Reservar memoria para el filtro en GPU: d_filter, la cual ya esta declarada
  // Copiar el filtro  (h_filter) a memoria global de la GPU (d_filter)
  checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(d_filterConst, h_filter, sizeof(float) * filterWidth * filterWidth, 0, cudaMemcpyHostToDevice));
}


void create_filter(float **h_filter, int *filterWidth){

  const int KernelWidth = KERNEL_SZ; //OJO CON EL TAMAÑO DEL FILTRO//
  *filterWidth = KernelWidth;

  //create and fill the filter we will convolve with
  *h_filter = new float[KernelWidth * KernelWidth];
  
  /*
  //Filtro gaussiano: blur
  const float KernelSigma = 2.;

  float filterSum = 0.f; //for normalization

  for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
    for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * KernelSigma * KernelSigma));
      (*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;

  for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
    for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
      (*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] *= normalizationFactor;
    }
  }
  */

#if FILTER == 2
  //Nitidez 5x5
  (*h_filter)[0] = 0;     (*h_filter)[1] = -1.;    (*h_filter)[2] = -1.;  (*h_filter)[3] = -1.;    (*h_filter)[4] = 0;
  (*h_filter)[5] = -1.;   (*h_filter)[6] = 2.;    (*h_filter)[7] = -4.;  (*h_filter)[8] = 2.;     (*h_filter)[9] = -1.;
  (*h_filter)[10] = -1.;  (*h_filter)[11] = -4.;  (*h_filter)[12] = 13.; (*h_filter)[13] = -4.;   (*h_filter)[14] = -1.;
  (*h_filter)[15] = -1.;  (*h_filter)[16] = 2.;   (*h_filter)[17] = -4.; (*h_filter)[18] = 2.;    (*h_filter)[19] = -1.;
  (*h_filter)[20] = 0;    (*h_filter)[21] = -1.;   (*h_filter)[22] = -1.; (*h_filter)[23] = -1.;    (*h_filter)[24] = 0;

#elif FILTER == 3
  //PasoAlto 5x5
  (*h_filter)[0] = 1.;   (*h_filter)[1] = 1.;   (*h_filter)[2] = 1.;    (*h_filter)[3] = 1.;    (*h_filter)[4] = 1.;
  (*h_filter)[5] = 1.;   (*h_filter)[6] = 4.;   (*h_filter)[7] = 4.;    (*h_filter)[8] = 4.;    (*h_filter)[9] = 1.;
  (*h_filter)[10] = 1.;  (*h_filter)[11] = 4.;  (*h_filter)[12] = 12.;  (*h_filter)[13] = 4.;   (*h_filter)[14] = 1.;
  (*h_filter)[15] = 1.;  (*h_filter)[16] = 4.;  (*h_filter)[17] = 4.;   (*h_filter)[18] = 4.;   (*h_filter)[19] = 1.;
  (*h_filter)[20] = 1.;  (*h_filter)[21] = 1.;  (*h_filter)[22] = 1.;   (*h_filter)[23] = 1.;   (*h_filter)[24] = 1.;

  for (int i = 0; i < 25; i++)
    (*h_filter)[i] /= 62.0;

#elif FILTER == 4
  //Media3x3
  (*h_filter)[0] = 1.;    (*h_filter)[1] = 1.;    (*h_filter)[2] = 1.;
  (*h_filter)[3] = 1.;    (*h_filter)[4] = 1.;    (*h_filter)[5] = 1.;
  (*h_filter)[6] = 1.;    (*h_filter)[7] = 1.;    (*h_filter)[8] = 1.;

  for (int i = 0; i < 9; i++)
    (*h_filter)[i] /= 9.0;

#elif FILTER == 5
  //Blur3x3
  (*h_filter)[0] = 1.;    (*h_filter)[1] = 2.;    (*h_filter)[2] = 1.;
  (*h_filter)[3] = 2.;    (*h_filter)[4] = 4.;    (*h_filter)[5] = 2.;  
  (*h_filter)[6] = 1.;    (*h_filter)[7] = 2.;    (*h_filter)[8] = 1.;

  for (int i = 0; i < 9; i++)
    (*h_filter)[i] /= 16.0;

#elif FILTER == 6
  //Blur5x5
  (*h_filter)[0] = 1.;   (*h_filter)[1] = 1.;   (*h_filter)[2] = 1.;    (*h_filter)[3] = 1.;    (*h_filter)[4] = 1.;
  (*h_filter)[5] = 1.;   (*h_filter)[6] = 4.;   (*h_filter)[7] = 4.;    (*h_filter)[8] = 4.;    (*h_filter)[9] = 1.;
  (*h_filter)[10] = 1.;  (*h_filter)[11] = 4.;  (*h_filter)[12] = 12.;  (*h_filter)[13] = 4.;   (*h_filter)[14] = 1.;
  (*h_filter)[15] = 1.;  (*h_filter)[16] = 4.;  (*h_filter)[17] = 4.;   (*h_filter)[18] = 4.;   (*h_filter)[19] = 1.;
  (*h_filter)[20] = 1.;  (*h_filter)[21] = 1.;  (*h_filter)[22] = 1.;   (*h_filter)[23] = 1.;   (*h_filter)[24] = 1.;

  for (int i = 0; i < 25; i++)
    (*h_filter)[i] /= 25.0;

#elif FILTER == 7
  const float KernelSigma = 2.;

  float filterSum = 0.f; //for normalization

  for (int r = -KernelWidth / 2; r <= KernelWidth / 2; ++r) {
    for (int c = -KernelWidth / 2; c <= KernelWidth / 2; ++c) {
      float filterValue = expf(-(float)(c * c + r * r) / (2.f * KernelSigma * KernelSigma));
      (*h_filter)[(r + KernelWidth / 2) * KernelWidth + c + KernelWidth / 2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;

  for (int r = -KernelWidth / 2; r <= KernelWidth / 2; ++r) {
    for (int c = -KernelWidth / 2; c <= KernelWidth / 2; ++c) {
      (*h_filter)[(r + KernelWidth / 2) * KernelWidth + c + KernelWidth / 2] *= normalizationFactor;
    }
  }

#elif FILTER == 8
  //SobelHorizontal3x3
  (*h_filter)[0] = -1.;   (*h_filter)[1] = -2.;   (*h_filter)[2] = -1.;
  (*h_filter)[3] = 0;     (*h_filter)[4] = 0;     (*h_filter)[5] = 0;
  (*h_filter)[6] = 1.;    (*h_filter)[7] = 2.;    (*h_filter)[8] = 1.;

#elif FILTER == 9
  //SobelVertical3x3
  (*h_filter)[0] = 1.;    (*h_filter)[1] = 2.;    (*h_filter)[2] = 1.;
  (*h_filter)[3] = 0;     (*h_filter)[4] = 0;     (*h_filter)[5] = 0;
  (*h_filter)[6] = -1.;   (*h_filter)[7] = -2.;   (*h_filter)[8] = -1.;

#else
  //Laplaciano 5x5
  (*h_filter)[0] = 0;   (*h_filter)[1] = 0;    (*h_filter)[2] = -1.;  (*h_filter)[3] = 0;    (*h_filter)[4] = 0;
  (*h_filter)[5] = 1.;  (*h_filter)[6] = -1.;  (*h_filter)[7] = -2.;  (*h_filter)[8] = -1.;  (*h_filter)[9] = 0;
  (*h_filter)[10] = -1.; (*h_filter)[11] = -2.; (*h_filter)[12] = 17.; (*h_filter)[13] = -2.; (*h_filter)[14] = -1.;
  (*h_filter)[15] = 1.; (*h_filter)[16] = -1.; (*h_filter)[17] = -2.; (*h_filter)[18] = -1.; (*h_filter)[19] = 0;
  (*h_filter)[20] = 1.;  (*h_filter)[21] = 0;   (*h_filter)[22] = -1.; (*h_filter)[23] = 0;   (*h_filter)[24] = 0;

#endif
}


void convolution(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redFiltered, 
                        unsigned char *d_greenFiltered, 
                        unsigned char *d_blueFiltered,
                        const int filterWidth)
{
  //TODO: Calcular tamaños de bloque
  const dim3 blockSize = {BLOCK_SZ, BLOCK_SZ, 1};
  const dim3 gridSize = { ((unsigned int)numCols-1)/blockSize.x+1, ((unsigned int)numRows-1)/blockSize.y+1, 1 };

  //TODO: Lanzar kernel para separar imagenes RGBA en diferentes colores
  separateChannels <<<gridSize, blockSize >>> (d_inputImageRGBA,
    numRows,
    numCols,
    d_red,
    d_green,
    d_blue
    );

  //TODO: Ejecutar convolución. Una por canal

#if SHARED == 1

  box_filter_shared <<<gridSize, blockSize, sizeof(unsigned char) * (blockSize.x + filterWidth - 1) * (blockSize.y + filterWidth - 1) >>> (
    d_red,
    d_redFiltered,
    numRows,
    numCols,
    d_filter,
    filterWidth
    );

  box_filter_shared <<<gridSize, blockSize, sizeof(unsigned char) * (blockSize.x + filterWidth - 1) * (blockSize.y + filterWidth - 1) >>> (
    d_green,
    d_greenFiltered,
    numRows,
    numCols,
    d_filter,
    filterWidth
    );

  box_filter_shared <<<gridSize, blockSize, sizeof(unsigned char) * (blockSize.x + filterWidth - 1) * (blockSize.y + filterWidth - 1) >>> (
    d_blue,
    d_blueFiltered,
    numRows,
    numCols,
    d_filter,
    filterWidth
    );


#else
  box_filter<<<gridSize, blockSize >>> (d_red,
    d_redFiltered,
    numRows,
    numCols,
    d_filter,
    filterWidth
    );

  box_filter << <gridSize, blockSize >> > (d_green,
    d_greenFiltered,
    numRows,
    numCols,
    d_filter,
    filterWidth
    );

  box_filter << <gridSize, blockSize >> > (d_blue,
    d_blueFiltered,
    numRows,
    numCols,
    d_filter,
    filterWidth
    );

#endif

  // Recombining the results. 
  recombineChannels<<<gridSize, blockSize>>>(d_redFiltered,
                                             d_greenFiltered,
                                             d_blueFiltered,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
}
