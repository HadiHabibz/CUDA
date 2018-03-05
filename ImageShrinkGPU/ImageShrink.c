#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BMP_HEADER_SIZE 54
#define blockSize 16


struct ImgProp
{
	int Hpixels;
	int Vpixels;
	unsigned char HeaderInfo[BMP_HEADER_SIZE];
	unsigned long int Hbytes;
};

struct ImgProp ip;
struct ImgProp ip2;
unsigned char * unprocessedImage;
unsigned char * processedImage;
int xFactor;
int yFactor;
double endTime;
double startTime;


// returns the time stamps in ms
double getTime()
{
    struct timeval tnow;

    gettimeofday( &tnow, NULL );
    return ( (double)tnow.tv_sec*1000000.0 + ( (double)tnow.tv_usec ) )/1000.00;
} // end of function getTime

unsigned char * readBMP( char* filename )
{
	unsigned char *TheImage;

	FILE * file = fopen( filename, "rb" );

	if( file == NULL )
	{
		printf( "%s NOT FOUND\n\n", filename );
		exit( EXIT_FAILURE );
	}

	unsigned char HeaderInfo[BMP_HEADER_SIZE];

	fread( HeaderInfo, sizeof( unsigned char ), BMP_HEADER_SIZE, file );

	// extract image height and width from header
	int width = *( int* )&HeaderInfo[18];
	int height = *( int* )&HeaderInfo[22];

	//copy header for re-use
	int i = 0;
	for(; i < BMP_HEADER_SIZE; i++ )
		ip.HeaderInfo[i] = HeaderInfo[i];

	ip.Vpixels = height;
	ip.Hpixels = width;

	int RowBytes = ( width*3 + 3 ) & ( ~3 );

	ip.Hbytes = RowBytes;

	printf("Input BMP File name: %20s  (%u x %u)\n", filename, ip.Hpixels, ip.Vpixels);

	TheImage = ( unsigned char * )malloc( ip.Vpixels * ip.Hbytes * sizeof(unsigned char*) );

	for(i = 0; i < ip.Vpixels * ip.Hbytes; i++)
		TheImage[i] = ( char )fgetc( file );

	fclose( file );

	return TheImage;
} // end of function readBMP


__global__ void shrinkKernel( unsigned char * input, unsigned char * output, int ipVpixels, int ipHbytes, int ip2Vpixels, int ip2Hbytes, int xFactor, int yFactor )
{
	int row;
	int col;

	row = blockIdx.x * blockDim.x + threadIdx.x;
	col = blockIdx.y * blockDim.y + threadIdx.y;

	if( row >= ip2Vpixels  || col >= ip2Hbytes-3 || row < 0 || col < 0 )
		return;

	output[ row * ip2Hbytes + col * 3 ] = input[ row * ipHbytes * yFactor + col * xFactor * 3 ];
	output[ row * ip2Hbytes + col * 3 + 1 ] = input[ row * ipHbytes * yFactor + col * xFactor * 3 + 1];
	output[ row * ip2Hbytes + col * 3 + 2 ] = input[ row * ipHbytes * yFactor + col * xFactor * 3 + 2];

} // end of kernel shrinkKernel

void launchGPU( void )
{
	unsigned char * gpuInput;
	unsigned char * gpuOutput;
	cudaError_t status;
	size_t unprocessedImageSize;
	size_t processedImageSize;

	ip2.Hpixels = ip.Hpixels / xFactor;
	ip2.Vpixels = ip.Vpixels / yFactor;
	ip2.Hbytes = ( ip2.Hpixels * 3 + 3 ) & ( ~3 );

	dim3 block( blockSize, blockSize, 1 );
	dim3 grid( ( ip2.Vpixels/block.x ) + 1, ( ip2.Hpixels/block.y ) + 1, 1 );

	unprocessedImageSize = ip.Hbytes * ip.Vpixels * sizeof( unsigned char );

	status = cudaMalloc( ( void ** ) &gpuInput, unprocessedImageSize );

	if( status != cudaSuccess )
	{
		printf( "Cannot allocate memory for gpuInput.\n" );
		exit( EXIT_FAILURE );
	}

	processedImageSize = ip2.Hbytes * ip2.Vpixels * sizeof( unsigned char );
	status = cudaMalloc( ( void ** ) &gpuOutput, processedImageSize );

	if( status != cudaSuccess )
	{
		printf( "Cannot allocate memory for gpuOutput.\n" );
		exit( EXIT_FAILURE );
	}

	status = cudaMemcpy( gpuInput, unprocessedImage, unprocessedImageSize, cudaMemcpyHostToDevice );

	if( status != cudaSuccess )
	{
		printf( "Cannot copy content to gpuInput.\n" );
		exit( EXIT_FAILURE );
	}

	startTime = getTime();
	//Kernel should be here
	shrinkKernel<<<grid, block>>>( gpuInput, gpuOutput, ip.Vpixels, ip.Hbytes, ip2.Vpixels, ip2.Hbytes, xFactor, yFactor );
	endTime = getTime();


	processedImage = ( unsigned char * ) malloc( processedImageSize );

	status = cudaMemcpy( processedImage, gpuOutput, processedImageSize, cudaMemcpyDeviceToHost );

	if( status != cudaSuccess )
	{
		printf( "Cannot copy content from gpuOutput.\n" );
		exit( EXIT_FAILURE );
	}

	cudaFree( gpuInput );
	cudaFree( gpuOutput );


} // end of function launchGPU

void writeBMP( unsigned char * img, char* filename )
{
	FILE * file = fopen( filename, "wb" );

	if( file == NULL )
	{
		printf( "FILE CREATION ERROR: %s\n\n",filename );
		exit( EXIT_FAILURE );
	}

	unsigned long int x;

	* ( ( int * ) &ip.HeaderInfo[2] ) = ( ip2.Hbytes * ip2.Vpixels + BMP_HEADER_SIZE );
	* ( ( int * ) &ip.HeaderInfo[18] ) = ip2.Hpixels;
	* ( ( int * ) &ip.HeaderInfo[22] ) = ip2.Vpixels;

	//write header
	for( x = 0; x < BMP_HEADER_SIZE; x++ )
		fputc( ip.HeaderInfo[x], file );

	for( x = 0; x < ip2.Vpixels * ip2.Hbytes; x++ )
		fputc( img[x], file );

	printf( "Output BMP File name: %20s  (%u x %u)\n", filename, ip2.Hpixels, ip2.Vpixels );

	fclose( file );
} // end of function writeBMP

int main( int argc, char ** argv )
{
	if( argc != 5 )
	{
		printf( "Usage: %s <input file> <output file> <x> <y>\n", argv[0] );
		exit( EXIT_FAILURE );
	}

	unprocessedImage = readBMP( argv[1] );
	xFactor = atoi( argv[3] );
	yFactor = atoi( argv[4] );
	launchGPU();
	writeBMP( processedImage, argv[2] );
	printf( "GPU Execution Time: %0.2f ms\n\n", (endTime - startTime) / 1.0 );
	free( processedImage );

} // end of function main
