#include "ImageStuff.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h>

double getTime()
{
    struct timeval tnow;

    gettimeofday( &tnow, NULL );
    return ( (double)tnow.tv_sec*1000000.0 + ( (double)tnow.tv_usec ) )/1000.00;
} // end of function getTime

struct ConstantValues
{
	double diagonal;
	double scaleFactor;
	double cosineScaled;
	double sineScaled;
	int xCenterOfImage;
	int yCenterOfImage;
}; // end of struct constantValues

void readCommandline( unsigned char ** originalImage, int * numberOfFrames, char * outputFilename, int argc, char ** argv )
{
	if( argc != 4 )
	{
		printf( "Usage: %s <input image> <output image> <number of rotations>\n", argv[0] );
		exit( EXIT_FAILURE );
	}

	*originalImage = readBMP( argv[1] );
	strcpy( outputFilename, argv[2] );
	*numberOfFrames = atoi( argv[3] );

	if( *numberOfFrames > 30 )
	{
		printf( "Number of frame should not exceed 30.\n" );
		exit( EXIT_FAILURE );
	}

	// Discard file type extension
	int i;
	for( i = 0; outputFilename[i] != '.'; i++ )
		; // empty loop

	outputFilename[i] = 0;

} // end of function readCommandline

void cleaning( unsigned char ** originalImage )
{
	free( *originalImage );
} // end of function cleaning

void checkCudaStatus( cudaError_t status )
{
	if( status == cudaSuccess )
		return;

	printf( "Error occurred in CUDA API" );
	exit( EXIT_FAILURE );
} // end of function checkCudaStatus

__global__ void rotateKernel( unsigned char * rotatedImage, const unsigned char * inputImage, const double angle,
		const int numberOfRows, const int numberOfCols, struct ConstantValues preCalculatedValues)
{
	const int blockSide = 32;

	__shared__ double sinY[blockSide];
	__shared__ double cosY[blockSide];
	__shared__ double variablesX[blockSide];

	int row;
	int col;
	int rowLocal;
	int colLocal;
	double pixelCartesianX;
	double pixelCartesianY;
	double sinPixelY;
	double cosPixelY;
	double newX;
	double newY;
	int newRow;
	int newCol;


	row = threadIdx.y + blockIdx.y * blockDim.y;
	col = threadIdx.x + blockIdx.x * blockDim.x;


	// Out of picture!
	if( row > numberOfRows || col*3 > numberOfCols )
		return;

	rowLocal = row % blockSide;
	colLocal = col % blockSide;

	if( colLocal == 0 )
	{
		pixelCartesianY = preCalculatedValues.yCenterOfImage - ( double ) row;
		sinPixelY = pixelCartesianY * preCalculatedValues.sineScaled;
		cosPixelY = pixelCartesianY * preCalculatedValues.cosineScaled;
		sinY[rowLocal] = sinPixelY;
		cosY[rowLocal] = cosPixelY;
	}

	__syncthreads();

	if( rowLocal == 0 )
	{
		pixelCartesianX = ( double ) ( col - preCalculatedValues.xCenterOfImage );
		variablesX[colLocal] = pixelCartesianX;
	}

	__syncthreads();

	newX = preCalculatedValues.cosineScaled * variablesX[colLocal] - sinY[rowLocal];
	newY = preCalculatedValues.sineScaled * variablesX[colLocal] + cosY[rowLocal];
	newRow = preCalculatedValues.yCenterOfImage - ( int ) newY;
	newCol = ( double ) ( preCalculatedValues.xCenterOfImage + newX );
	col *= 3;

	if( newCol >= 0 && newRow >= 0 && newCol < numberOfCols && newRow < numberOfRows )
	{
		newCol *= 3;
		rotatedImage[newRow * numberOfCols + newCol] = inputImage[row * numberOfCols + col];
		rotatedImage[newRow * numberOfCols + newCol + 1] = inputImage[row * numberOfCols + col + 1];
		rotatedImage[newRow * numberOfCols + newCol + 2] = inputImage[row * numberOfCols + col + 2];
	}
} // end of kernel rotateKernel

// THIS IS WITH NO DIVERGENCE. MUCH BETTER THAN THE ONE ABOVE
/*__global__ void rotateKernel( unsigned char * rotatedImage, const unsigned char * inputImage, const double angle,
		const int numberOfRows, const int numberOfCols, struct ConstantValues preCalculatedValues)
{
	const int blockSideX = 32;
	const int blockSideY = 32;

	__shared__ double sinY[blockSideY];
	__shared__ double cosY[blockSideY];
	__shared__ double variablesX[blockSideX];

	int row;
	int col;
	int rowLocal;
	int colLocal;
	double pixelCartesianX;
	double pixelCartesianY;
	double sinPixelY;
	double cosPixelY;
	double newX;
	double newY;
	int newRow;
	int newCol;


	row = threadIdx.y + blockIdx.y * blockDim.y;
	col = threadIdx.x + blockIdx.x * blockDim.x;


	// Out of picture!
	if( row > numberOfRows || col*3 > numberOfCols )
		return;

	rowLocal = row % blockSideY;
	colLocal = col % blockSideX;

	//if( colLocal == 0 )
	//{
	//	pixelCartesianY = preCalculatedValues.yCenterOfImage - ( double ) row;
	//	sinPixelY = pixelCartesianY * preCalculatedValues.sineScaled;
	//	cosPixelY = pixelCartesianY * preCalculatedValues.cosineScaled;
	//	sinY[rowLocal] = sinPixelY;
	//	cosY[rowLocal] = cosPixelY;
	//}

	//__syncthreads();

	if( rowLocal == 0 )
	{
		pixelCartesianY = preCalculatedValues.yCenterOfImage - ( double ) (colLocal + blockIdx. y * blockDim.y);
		sinPixelY = pixelCartesianY * preCalculatedValues.sineScaled;
		cosPixelY = pixelCartesianY * preCalculatedValues.cosineScaled;
		sinY[colLocal] = sinPixelY;
		cosY[colLocal] = cosPixelY;
		pixelCartesianX = ( double ) ( col - preCalculatedValues.xCenterOfImage );
		variablesX[colLocal] = pixelCartesianX;
	}

	__syncthreads();

	newX = preCalculatedValues.cosineScaled * variablesX[colLocal] - sinY[rowLocal];
	newY = preCalculatedValues.sineScaled * variablesX[colLocal] + cosY[rowLocal];
	newRow = preCalculatedValues.yCenterOfImage - ( int ) newY;
	newCol = ( double ) ( preCalculatedValues.xCenterOfImage + newX );
	col *= 3;

	if( newCol >= 0 && newRow >= 0 && newCol < numberOfCols && newRow < numberOfRows )
	{
		newCol *= 3;
		rotatedImage[newRow * numberOfCols + newCol] = inputImage[row * numberOfCols + col];
		rotatedImage[newRow * numberOfCols + newCol + 1] = inputImage[row * numberOfCols + col + 1];
		rotatedImage[newRow * numberOfCols + newCol + 2] = inputImage[row * numberOfCols + col + 2];
	}
} // end of kernel rotateKernel
*/
void calculateConstantValues( struct  ConstantValues * constantValues, const double rotateAngle )
{
	const int numberOfRows = ip.Vpixels;
	const int numberOfCols = ip.Hpixels;

	constantValues->diagonal = sqrt ( ( double ) ( numberOfRows * numberOfRows + numberOfCols * numberOfCols ) );
	constantValues->scaleFactor = ( numberOfCols > numberOfRows ) ? ( ( double ) numberOfRows ) / constantValues->diagonal : ( ( double ) numberOfCols ) / constantValues->diagonal;
	constantValues->cosineScaled = cos( rotateAngle ) * constantValues->scaleFactor;
	constantValues->sineScaled = sin( rotateAngle ) * constantValues->scaleFactor;
	constantValues->xCenterOfImage = numberOfCols / 2;
	constantValues->yCenterOfImage = numberOfRows / 2;

} // end of function calculateConstantValues

double launchGPU( const unsigned char * originalImage, unsigned char * resultImage, const double angle, const int numberOfFrames,
		const char * outputFilename, const int i )
{
	const int maximumStringSize = 50;
	const int blockSize = 32;
	const int height = ip.Vpixels;
	const int width = ip.Hbytes;
	const int backgroundColor = 0;

	cudaError status;
	unsigned char * gpuInput;
	unsigned char * gpuResult;
	int gpuArraySize;
	dim3 grid;
	dim3 block;
	char filename[maximumStringSize];
	char appendix[2];
	double currentAngle;
	struct ConstantValues preCalculatedVals;
	double end;
	double start;

	block = dim3( blockSize, blockSize );
	grid = dim3( ( width / blockSize ) + 1, ( height / blockSize ) + 1 );
	gpuArraySize = height * width * sizeof( unsigned char );
	currentAngle = i * angle;

	status = cudaSetDevice( 0 );
	checkCudaStatus( status );

	status = cudaMalloc( ( void ** ) &gpuInput, gpuArraySize );
	checkCudaStatus( status );

	status = cudaMalloc( ( void ** ) &gpuResult, gpuArraySize );
	checkCudaStatus( status );

	status = cudaMemcpy( gpuInput, originalImage, gpuArraySize, cudaMemcpyHostToDevice );
	checkCudaStatus( status );

	status = cudaMemset( gpuResult, backgroundColor, gpuArraySize );
	checkCudaStatus( status );

	calculateConstantValues( &preCalculatedVals, currentAngle );

	start = getTime();
	rotateKernel<<< grid, block >>>( gpuResult, gpuInput, currentAngle, height, width, preCalculatedVals );
	cudaDeviceSynchronize();
	end = getTime();

	status = cudaMemcpy( resultImage, gpuResult, gpuArraySize, cudaMemcpyDeviceToHost );
	checkCudaStatus( status );

	strcpy( filename, outputFilename );
	sprintf( appendix, "%02d", i);
	strcat( filename, appendix );
	strcat( filename, ".bmp" );

	writeBMP( resultImage, filename );

	cudaFree( gpuInput );
	cudaFree( gpuResult );

	return ( end - start );

} // end of function launchGPU

void rotateNTimes( const unsigned char * originalImage, const int numberOfFrames, const char * outputFilename )
{
	int rotationDegree;
	double rotationAngleRadian;
	const double pi = 3.141592;
	unsigned char * rotatedImaage;
	double totalTime;

	totalTime = 0;
	rotationDegree = 360 / numberOfFrames;
	rotationAngleRadian =  pi / 180.0  * ( double ) rotationDegree;

	rotatedImaage = ( unsigned char * ) malloc( ip.Hbytes * ip.Vpixels * sizeof( unsigned char ) );

	if( rotatedImaage == NULL )
	{
		printf( "Failed to allocated memory for rotated image.\n" );
		exit( EXIT_FAILURE );
	}

	for( int i = 0; i < numberOfFrames; i++ )
		totalTime += launchGPU( originalImage, rotatedImaage, rotationAngleRadian, numberOfFrames, outputFilename, i );

	printf( "-------------------------------------------------------------------------------------------------------\n" );
	printf( "Total Kernel Execution Time: %.3lf ms\n", totalTime );
	printf( "Average Kerenl Execution Time: %.3lf ms/frame \n", totalTime / ( double ) numberOfFrames );
	printf( "-------------------------------------------------------------------------------------------------------\n" );

} // end of function rotateNTimes


int main( int argc, char ** argv )
{
	const int stringSizeMax = 50;
	unsigned char * originalImage;
	int numberOfFrames;
	char outputFilename[stringSizeMax];

	readCommandline( &originalImage, &numberOfFrames, outputFilename, argc, argv );
	rotateNTimes( originalImage, numberOfFrames, outputFilename );

	cleaning( &originalImage );
	printf( "Done!" );
	return 0;
} // end of function main
