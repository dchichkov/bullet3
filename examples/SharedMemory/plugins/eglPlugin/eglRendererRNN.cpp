/*
eglRendererRNN
Copyright (c) 2018 Dmitry Chichkov

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use
of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim
that you wrote the original software. If you use this software in a product, an
acknowledgment in the product documentation would be appreciated but is not
required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>

//#define DEBUG_RNN_INFERENCE
#ifdef DEBUG_RNN_INFERENCE
#include <algorithm>
#include "stb_image/stb_image_write.h"
#endif  // DEBUG_RNN_INFERENCE


// Error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      b3Error("CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cudnnErrCheck(stat) { cudnnErrCheck_((stat), __FILE__, __LINE__); }
void cudnnErrCheck_(cudnnStatus_t stat, const char *file, int line) {
   if (stat != CUDNN_STATUS_SUCCESS) {
      b3Error(stderr, "cuDNN Error: %s %s %d\n", cudnnGetErrorString(stat), file, line);
   }
}


/*
__global__ void initGPUData_ker(float *data, int numElements, float value) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid < numElements) {
      data[tid] = value;
   }
}

void initGPUData(float *data, int numElements, float value) {
   dim3 gridDim;
   dim3 blockDim;

   blockDim.x = 1024;
   gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

   initGPUData_ker <<< gridDim, blockDim >>> (data, numElements, value);
}*/


void initGPUData(float *data, int numElements, float value) {
}


struct EGLRendererRNN
{
	/** \brief Initializes a RNN inference engine from .uff or .plan model files.
	* \param modelFileName a path to .uff or .plan model
	* \param modelInputLayer a name of a input layer. Should be float32, in
	* range of 0..1, shaped (height, width, 3).
	* \param modelOutputLayers a list of output layer names, trailing with zero.
	* Supports only float32 layer outputs.
	* \param width expected width of the input layer (and image).
	* \param height expected height of the input layer.
	* \param kBatchSize process a batch of width x height images. Expected
	* rendered input is width x (height*kBatchSize).
	* This initializes a RNN inference engine from .uff or .plan model files.
	* Please refer to RNN documentation and examples to create such files.
	*
	* Hint: Use uff.from_tensorflow() and uff_to_plan to convert TensorFlow
	* models.
	*
	* Usage:
	*/
	EGLRendererRNN(const char *modelFileName, const char *modelInputLayer,
						const char **modelOutputLayers, int width, int height,
						int kBatchSize = 1, int kWorkspaceSize = 1 << 26);

	~EGLRendererRNN();

	int m_width, m_height;
	int m_kBatchSize;

	/// output size in bytes (incl. batch calc.)
	int m_totalOutputSize;

	// feature length in floats
	int m_featureLength;

	/// PBO (pixels to attach to RNN)
	unsigned int pbo;
	cudaGraphicsResource_t pboRes;

	/// CUDA / RNN memory, used to store output layer before transferring to CPU
	void *outputDataDevice;


	// Tensor descriptors
	btAlignedObjectArray<cudnnTensorDescriptor_t>  xDesc, yDesc;
   	cudnnTensorDescriptor_t hxDesc, cxDesc;
   	cudnnTensorDescriptor_t hyDesc, cyDesc;


	// CUDA / RNN engine memory bindings, contains CUDA pointers indexed by
	// binding indexes
	btAlignedObjectArray<void *> bindings;

	// Index into bindings array (for the input layer)
	int m_inputBindingIndex;


	int getFeatureLength()
	{
		return m_featureLength;
	}

	void uninitRNNEngine();

	/** \brief Transfers pixels from GL to CUDA, executes RNN engine and
	* outputs the result to outputBuffer.
	* \param outputBuffer, a pointer to CPU memory
	* \param outputBufferSizeInBytes buffer size in bytes.
	*
	*/
	size_t copyCameraImageFeatures(float *outputBuffer,
								   size_t outputBufferSizeInBytes);


	/** Returns tensor size, in elements.
	*/
	static size_t size(nvinfer1::Dims shape)
	{
		size_t size = shape.nbDims > 0 ? 1 : 0;
		for (int i = 0; i < shape.nbDims; i++)
			size *= shape.d[i];
		return size;
	}
};

EGLRendererRNN::EGLRendererRNN(const char *modelFileName,
											  const char *modelInputLayer,
											  const char **modelOutputLayers,
											  int width, int height,
											  int kBatchSize, int kWorkspaceSize)
	: m_width(width), m_height(height), m_kBatchSize(kBatchSize), m_totalOutputSize(0), m_featureLength(0),
	  pbo(0), pboRes(0), outputDataDevice(0), engine(0), context(0), m_inputBindingIndex(0),
	  kPopulation(16),
	  seqLength(height / 2),  		// 80
      numLayers(2),
      hiddenSize(width * 3 * 2), 	// 960
      inputSize(width * 3 * 2),		// 960
      miniBatch(kBatchSize * kPopulation),
      dropout(0),
      bidirectional(0),
      mode(0),						// RNN_RELU
      persistent(0)

{

   // Memory allocation. hx, cx, dhx, dcx, hy, cy, dhy and dcy can be NULL.

   cudaErrCheck(cudaMalloc((void**)&x, seqLength * inputSize * miniBatch * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&hx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&cx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&y, seqLength * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&hy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&cy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1) * sizeof(float)));


   // Set up tensor descriptors. x/y/dx/dy are arrays, one per time step.
   xDesc.resize(seqLength);
   yDesc.resize(seqLength);


   // In this example dimA[1] is constant across the whole sequence
   // This isn't required, all that is required is that it does not increase.
   for (int i = 0; i < seqLength; i++) {
      cudnnErrCheck(cudnnCreateTensorDescriptor(&xDesc[i]));
      cudnnErrCheck(cudnnCreateTensorDescriptor(&yDesc[i]));

      dimA[0] = miniBatch;
      dimA[1] = inputSize;
      dimA[2] = 1;

      strideA[0] = dimA[2] * dimA[1];
      strideA[1] = dimA[2];
      strideA[2] = 1;

      cudnnErrCheck(cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));

      dimA[0] = miniBatch;
      dimA[1] = bidirectional ? hiddenSize * 2 : hiddenSize;
      dimA[2] = 1;

      strideA[0] = dimA[2] * dimA[1];
      strideA[1] = dimA[2];
      strideA[2] = 1;

      cudnnErrCheck(cudnnSetTensorNdDescriptor(yDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
   }


   dimA[0] = numLayers * (bidirectional ? 2 : 1);
   dimA[1] = miniBatch;
   dimA[2] = hiddenSize;

   strideA[0] = dimA[2] * dimA[1];
   strideA[1] = dimA[2];
   strideA[2] = 1;

   cudnnErrCheck(cudnnCreateTensorDescriptor(&hxDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&cxDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&hyDesc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&cyDesc));

   cudnnErrCheck(cudnnSetTensorNdDescriptor(hxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(cxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(hyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(cyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));


   // -------------------------
   // Set up the dropout descriptor (needed for the RNN descriptor, no dropout)
   // -------------------------
   cudnnDropoutDescriptor_t dropoutDesc;
   cudnnErrCheck(cudnnCreateDropoutDescriptor(&dropoutDesc));
   cudnnErrCheck( cudnnSetDropoutDescriptor(dropoutDesc, 
                     cudnnHandle, 0 /* dropout */, 0, 0 /* state_size */, 0 /* seed */) );


   // -------------------------
   // Set up the RNN descriptor
   // -------------------------
   cudnnRNNDescriptor_t rnnDesc;
   cudnnRNNMode_t RNNMode;
   cudnnRNNAlgo_t RNNAlgo;

   cudnnErrCheck(cudnnCreateRNNDescriptor(&rnnDesc));

   if      (mode == 0) RNNMode = CUDNN_RNN_RELU;
   else if (mode == 1) RNNMode = CUDNN_RNN_TANH;
   else if (mode == 2) RNNMode = CUDNN_LSTM;
   else if (mode == 3) RNNMode = CUDNN_GRU;

   // Persistent RNNs are only supported on Pascal+ GPUs.
   if      (persistent == 0) RNNAlgo = CUDNN_RNN_ALGO_STANDARD;
   else if (persistent == 1) RNNAlgo = CUDNN_RNN_ALGO_PERSIST_STATIC;
   else if (persistent == 2) RNNAlgo = CUDNN_RNN_ALGO_PERSIST_DYNAMIC;

   cudnnErrCheck(cudnnSetRNNDescriptor_v6(cudnnHandle,
                                       rnnDesc,
                                       hiddenSize,
                                       numLayers,
                                       dropoutDesc,
                                       CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
                                       bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
                                       RNNMode,
                                       RNNAlgo, // Can be changed to use persistent RNNs on Pascal+ GPUs.
                                       CUDNN_DATA_FLOAT));


   // Set the math type to allow cuDNN to use Tensor Cores:
   // cudnnErrCheck ( cudnnSetRNNMatrixMathType(rnnDesc, CUDNN_TENSOR_OP_MATH) );

   // -------------------------
   // Set up parameters
   // -------------------------
   // This needs to be done after the rnn descriptor is set as otherwise
   // we don't know how many parameters we have to allocate
   void *w;
   void *dw;

   cudnnFilterDescriptor_t wDesc, dwDesc;

   cudnnErrCheck(cudnnCreateFilterDescriptor(&wDesc));
   cudnnErrCheck(cudnnCreateFilterDescriptor(&dwDesc));

   size_t weightsSize;
   cudnnErrCheck(cudnnGetRNNParamsSize(cudnnHandle, rnnDesc, xDesc[0], &weightsSize, CUDNN_DATA_FLOAT));

   int dimW[3];
   dimW[0] =  weightsSize / sizeof(float);
   dimW[1] = 1;
   dimW[2] = 1;

   cudnnErrCheck(cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));
   cudnnErrCheck(cudnnSetFilterNdDescriptor(dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));

   cudaErrCheck(cudaMalloc((void**)&w,  weightsSize));
   cudaErrCheck(cudaMalloc((void**)&dw, weightsSize));


   // -------------------------
   // Set up work space and reserved memory
   // -------------------------
   void *workspace;
   void *reserveSpace;

   size_t workSize;
   size_t reserveSize;

   // Need for every pass
   cudnnErrCheck(cudnnGetRNNWorkspaceSize(cudnnHandle, rnnDesc, seqLength, xDesc, &workSize));
   // Only needed in training, shouldn't be touched between passes.
   cudnnErrCheck(cudnnGetRNNTrainingReserveSize(cudnnHandle, rnnDesc, seqLength, xDesc, &reserveSize));

   cudaErrCheck(cudaMalloc((void**)&workspace, workSize));
   cudaErrCheck(cudaMalloc((void**)&reserveSpace, reserveSize));

   // *********************************************************************************************************
   // Initialise weights and inputs
   // *********************************************************************************************************
   // We initialise to something simple.
   // Matrices are initialised to 1 / matrixSize, biases to 1, data is 1.
   initGPUData((float*)x, seqLength * inputSize * miniBatch, 1.f);
   if (hx != NULL) initGPUData((float*)hx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.f);
   if (cx != NULL) initGPUData((float*)cx, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.f);

   initGPUData((float*)dy, seqLength * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.f);
   if (dhy != NULL) initGPUData((float*)dhy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.f);
   if (dcy != NULL) initGPUData((float*)dcy, numLayers * hiddenSize * miniBatch * (bidirectional ? 2 : 1), 1.f);


   // Weights
   int numLinearLayers = 0;
   if (RNNMode == CUDNN_RNN_RELU || RNNMode == CUDNN_RNN_TANH) {
      numLinearLayers = 2;
   }
   else if (RNNMode == CUDNN_LSTM) {
      numLinearLayers = 8;
   }
   else if (RNNMode == CUDNN_GRU) {
      numLinearLayers = 6;
   }

   for (int layer = 0; layer < numLayers * (bidirectional ? 2 : 1); layer++) {
      for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
         cudnnFilterDescriptor_t linLayerMatDesc;
         cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerMatDesc));
         float *linLayerMat;

         cudnnErrCheck(cudnnGetRNNLinLayerMatrixParams( cudnnHandle,
                                                        rnnDesc,
                                                        layer,
                                                        xDesc[0],
                                                        wDesc,
                                                        w,
                                                        linLayerID,
                                                        linLayerMatDesc,
                                                        (void**)&linLayerMat));

         cudnnDataType_t dataType;
         cudnnTensorFormat_t format;
         int nbDims;
         int filterDimA[3];
         cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerMatDesc,
                                                  3,
                                                  &dataType,
                                                  &format,
                                                  &nbDims,
                                                  filterDimA));

         initGPUData(linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f / (float)(filterDimA[0] * filterDimA[1] * filterDimA[2]));

         cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerMatDesc));

         cudnnFilterDescriptor_t linLayerBiasDesc;
         cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
         float *linLayerBias;

         cudnnErrCheck(cudnnGetRNNLinLayerBiasParams( cudnnHandle,
                                                        rnnDesc,
                                                        layer,
                                                        xDesc[0],
                                                        wDesc,
                                                        w,
                                                        linLayerID,
                                                        linLayerBiasDesc,
                                                        (void**)&linLayerBias));

         cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerBiasDesc,
                                                  3,
                                                  &dataType,
                                                  &format,
                                                  &nbDims,
                                                  filterDimA));

         initGPUData(linLayerBias, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f);

         cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
      }
   }

   // *********************************************************************************************************
   // Dynamic persistent RNN plan (if using this algo)
   // *********************************************************************************************************
   cudnnPersistentRNNPlan_t rnnPlan;
   if (RNNAlgo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
      // Note: This step is expensive. Once completed the plan can be reused so long as the descriptor
      //       minibatch or datatype don't change.
      cudnnErrCheck(cudnnCreatePersistentRNNPlan(rnnDesc, miniBatch, CUDNN_DATA_FLOAT, &rnnPlan));
      // Tell calls using this descriptor which plan to use.
      cudnnErrCheck(cudnnSetPersistentRNNPlan(rnnDesc, rnnPlan));
   }

   // *********************************************************************************************************
   // At this point all of the setup is done. We now need to pass through the RNN.
   // *********************************************************************************************************
   cudaErrCheck(cudaDeviceSynchronize());

   cudaEvent_t start, stop;
   float timeForward;
   cudaErrCheck(cudaEventCreate(&start));
   cudaErrCheck(cudaEventCreate(&stop));

   cudaErrCheck(cudaEventRecord(start));

   // Run inference
   cudnnErrCheck(cudnnRNNForwardInference(cudnnHandle,
                                         rnnDesc,
                                         seqLength,
                                         xDesc,
                                         x,
                                         hxDesc,
                                         hx,
                                         cxDesc,
                                         cx,
                                         wDesc,
                                         w,
                                         yDesc,
                                         y,
                                         hyDesc,
                                         hy,
                                         cyDesc,
                                         cy,
                                         workspace,
                                         workSize));


   cudaErrCheck(cudaEventRecord(stop));
   cudaErrCheck(cudaEventSynchronize(stop));
   cudaErrCheck(cudaEventElapsedTime(&timeForward, start, stop));

   // Calculate FLOPS
   printf("Forward: %3.0f ms, %3.0f GFLOPS\n", timeForward, numMats * 2ull * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeForward));


	/*
	if (builder == 0)
	{
		b3Error(
			"Failed to create RNN Builder object, please check your "
			"RNN installation or attempt to load .plan file.\n");
		return;
	}


	// make sure that that rendering is the same size as the network input
	Dims inputDims = engine->getBindingDimensions(m_inputBindingIndex);
	if (m_width != inputDims.d[1] || m_height != inputDims.d[2] ||
		3 != inputDims.d[0])
	{
		b3Error(
			"Error rendered image is %d x %d x %d and inference engine expects "
			"%d x %d x %d.\n",
			m_width, m_height, 3, inputDims.d[1], inputDims.d[2],
			inputDims.d[0]);
		uninitRNNEngine();
		return;
	}

	bindings.resize(m_inputBindingIndex + 1);

	// Allocate CUDA memory for output buffer
	cudaMalloc(&outputDataDevice, m_totalOutputSize);
	if (outputDataDevice == 0)
	{
		b3Error(
			"Failed to allocate %d bytes of CUDA memory for the RNN "
			"engine. Please make sure sufficient CUDA memory is available.\n",
			m_totalOutputSize);
		uninitRNNEngine();
		return;
	}

	*/
	// we need to initialize PBO, but can't do it yet, because GL is not yet
	// ready.
	pbo = 0;
}

void EGLRendererRNN::uninitRNNEngine()
{	
	if (outputDataDevice)
	{
		cudaFree(outputDataDevice);
		outputDataDevice = 0;
	}
}

EGLRendererRNN::~EGLRendererRNN() { uninitRNNEngine(); }

size_t
EGLRendererRNN::copyCameraImageFeatures(float *outputBuffer,
											 size_t outputBufferSizeInBytes)
{
	if (m_totalOutputSize > outputBufferSizeInBytes)
	{
		b3Error(
			"Error during rendering-inferencing, CPU buffer size to store "
			"output features is too small. Expected %d, provided %d\n",
			m_totalOutputSize, outputBufferSizeInBytes);
		return 0;
	}

	// Bind buffer (or Generate PBO, if it is a first run)
	if (!pbo)
	{
		glGenBuffers(1, &pbo);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
		glBufferData(GL_PIXEL_PACK_BUFFER,
					 3 * (m_kBatchSize * m_width * m_height * sizeof(GLfloat)), NULL,
					 GL_DYNAMIC_READ);

		if (glGetError() != GL_NO_ERROR)
		{
			b3Error(
				"Error during rendering-inferencing, rendered image size is "
				"invalid (?) (should be %dx%dx3).\n",
				m_height * m_kBatchSize, m_width);
			return 0;
		}

		// Register buffer to CUDA memory
		if (cudaGraphicsGLRegisterBuffer(&pboRes, pbo,
										 cudaGraphicsRegisterFlagsReadOnly) != 0)
		{  // cudaGraphicsMapFlagsWriteDiscard
			b3Error(
				"Error during registering GL buffer in CUDA, rendered image size "
				"is invalid (?) (should be %dx%dx3).\n",
				m_height * m_kBatchSize, m_width);
			return 0;
		}
	}
	else
	{
		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
		glBufferData(GL_PIXEL_PACK_BUFFER,
					 3 * (m_kBatchSize * m_width * m_height * sizeof(GLfloat)), NULL,
					 GL_DYNAMIC_READ);
		if (glGetError() != GL_NO_ERROR)
		{
			b3Error(
				"Error during rendering-inferencing, rendered image size is "
				"invalid (should be %dx%dx3).",
				m_height * m_kBatchSize, m_width);
			return 0;
		}
	}

	// NOTE, we are casting to GL_FLOAT, because RNN accepts only floats,
	// TODO: use bytes.
	glReadPixels(0, 0, m_width, m_height * m_kBatchSize, GL_RGB, GL_FLOAT,
				 0);  // 0 is an *offset* into the buffer, not the pointer

	if (glGetError() != GL_NO_ERROR)
	{
		b3Error(
			"Error during rendering-inferencing, rendered image size is "
			"invalid (should be %dx%dx3).",
			m_height * m_kBatchSize, m_width);
		return 0;
	}

	// Unbind the buffer from PBO
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	// Map PBO to CUDA
	cudaGraphicsMapResources(1, &pboRes);

	// Obtain CUDA pointer of PBO. NOTE: FLOATs, range 0..1.  :(  If only RNN
	// could accept bytes
	void *inputDataDevice = NULL;
	size_t size = 0;
	cudaGraphicsResourceGetMappedPointer(&inputDataDevice, &size, pboRes);
	if (inputDataDevice == 0)
	{
		b3Error(
			"Error during rendering-inferencing, failure in "
			"cudaGraphicsResourceGetMappedPointer. Try different CUDA/driver "
			"version(?).");
		return 0;
	}

	// Fill RNN device bindings, see m_inputBindingIndex. outputBindingIndexes
	// are already set in the init
	bindings[m_inputBindingIndex] = (void *)inputDataDevice;

	// Run Inference
   // If we're not training we use this instead
   cudnnErrCheck(cudnnRNNForwardInference(cudnnHandle,
                                         rnnDesc,
                                         seqLength,
                                         xDesc,
                                         x,
                                         hxDesc,
                                         hx,
                                         cxDesc,
                                         cx,
                                         wDesc,
                                         w,
                                         yDesc,
                                         y,
                                         hyDesc,
                                         hy,
                                         cyDesc,
                                         cy,
                                         workspace,
                                         workSize));





	if (!context->execute(m_kBatchSize, &bindings[0]))
	{
		b3Error(
			"Error during rendering-inferencing, failure executing RNN "
			"engine. See RNN log, try different RNN version(?).");
		return 0;
	}

#ifdef DEBUG_RNN_INFERENCE
	// transfer input image back to host and save it (debug)
	float *inputDataHost =
		(float *)malloc(3 * (m_width * m_height * sizeof(GLfloat)));
	memset(inputDataHost, 0, 3 * (m_width * m_height * sizeof(GLfloat)));
	cudaMemcpy(inputDataHost, inputDataDevice,
			   3 * (m_width * m_height * sizeof(GLfloat)),
			   cudaMemcpyDeviceToHost);

	unsigned char *rgbaBuffer =
		(unsigned char *)malloc(3 * (m_width * m_height * sizeof(unsigned char)));

	for (int i = 0; i < 3 * m_width * m_height; i++)
	{
		rgbaBuffer[i] =
			(unsigned char)(inputDataHost[i] *
							255.0);  // FLOAT, range 0..1, converting to bytes
	}
	stbi_write_png("getScreenPixels.dump.png", m_width, m_height, 3, rgbaBuffer,
				   m_width * 3);

	// inputDataHost: 0.890196 0.835294 0.603922 0.694118 0.341176
	printf("inputDataHost: %f %f %f %f %f\n", inputDataHost[0],
		   inputDataHost[m_width * m_height + 100],
		   inputDataHost[m_width * m_height + 101],
		   inputDataHost[m_width * m_height + 102],
		   inputDataHost[3 * m_width * m_height - 1]);
	free(inputDataHost);
	free(rgbaBuffer);
#endif  // DEBUG_RNN_INFERENCE

	// Unmap resources
	cudaGraphicsUnmapResources(1, &pboRes);

	// Transfer RNN output to CPU
	cudaMemcpy(outputBuffer, outputDataDevice, m_totalOutputSize,
			   cudaMemcpyDeviceToHost);

#ifdef DEBUG_RNN_INFERENCE
	// copy output to host and print activations (useful for debugging)
	float *outputDataHost = (float *)malloc(m_totalOutputSize);
	memset(outputDataHost, 0, m_totalOutputSize);
	cudaMemcpy(outputDataHost, outputDataDevice, m_totalOutputSize,
			   cudaMemcpyDeviceToHost);

	// 905:0.520552 633:0.028283 808:0.017819 316:0.015341 906:0.014210
	printf("\n");
	for (int i = 0; i < 5; i++)
	{
		int max_i = std::max_element(outputDataHost, outputDataHost + m_totalOutputSize / sizeof(float)) -
					outputDataHost;
		printf("%d:%f ", max_i, outputDataHost[max_i]);
		outputDataHost[max_i] = 0;
	}
	printf("\n");
	free(outputDataHost);
#endif  // DEBUG_RNN_INFERENCE
}
