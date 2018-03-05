#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// returns the time stamps in ms
double getTime()
{
    struct timeval tnow;

    gettimeofday( &tnow, NULL );
    return ( (double)tnow.tv_sec*1000000.0 + ( (double)tnow.tv_usec ) )/1000.00;
} // end of function getTime

void readCommandline( int argc, char ** argv, int * howmany, char * type, int * chunkSize, int * blockSize )
{
	if( argc != 5 )
	{
		printf( "Usage: %s <How many> <type> <chunk size> <block size>\n", argv[0] );
		exit( EXIT_SUCCESS );
	}

	*howmany = atoi( argv[1] );
	*howmany = *howmany * 1024 * 1024;

	*type = ( char ) toupper( ( char ) argv[2][0] );
	*chunkSize = atoi ( argv[3] );
	*blockSize = atoi( argv[4] );
} // end of function readCommandline

void * createArray( const int length, const char type )
{
	int elementSize;
	void * ary;

	switch( type )
	{
	case 'I':
		elementSize = sizeof( int );
		break;

	case 'L':
		elementSize = sizeof( long int );
		break;

	case 'F':
		elementSize = sizeof( float );
		break;

	case 'D':
		elementSize = sizeof( double );
		break;

	default:
		printf( "Invalid data type.\n" );
		exit( EXIT_SUCCESS );
	}

	ary = malloc( length * elementSize );

	if( ary == NULL )
	{
		printf( "Could not allocate memory.\n" );
		exit( EXIT_FAILURE );
	}

	memset( ary, 0, length * elementSize );

	return ary;

} // end of function createArray

void houseKeeping( void * ary1, void * ary2 )
{
	free( ary1 );
	free( ary2 );
} // end of function houseKeeping

void doubleRandomInitializer( double * ary, const int size )
{
	int i = 0;

	for( i = 0; i < size; i ++ )
		ary[i] = rand() * 1.0;

} // end of function doubleRandomInitializer

void floatRandomInitializer( float * ary, const int size )
{
	int i = 0;

	for( i = 0; i < size; i ++ )
		ary[i] = rand() * 1.0;

} // end of function floatRandomInitializer

void intRandomInitializer( int * ary, const int size )
{
	int i = 0;

	for( i = 0; i < size; i ++ )
		ary[i] = rand();

} // end of function intRandomInitializer

void longRandomInitializer( long * ary, const int size )
{
	int i = 0;
	int temp;

	for( i = 0; i < size; i ++ ){
		 temp =  ( long ) rand();
		 ary[i] = temp; }

} // end of function lonRandomInitializer

void randomInitializer( void * ary, const int length, const char type )
{
	switch( type )
	{
	case 'I':
		intRandomInitializer( ( int * ) ary, length );
		break;

	case 'L':
		longRandomInitializer( ( long * ) ary, length );
		break;

	case 'F':
		floatRandomInitializer( ( float * ) ary, length );
		break;

	case 'D':
		doubleRandomInitializer( ( double * ) ary, length );
		break;

	default:
		printf( "Failed to initialize the array. Cannot recognize its type.\n" );
		exit( EXIT_FAILURE );
	}
} // end of function randomInitializer

void mergeInt( int * sorted, const int * unsorted, const int length, const int chunkSize )
{
	int numberOfChunks;
	int * index;
	int min;
	int minIndex;
	int i;

	numberOfChunks = ceil( ( double )length / chunkSize );

	index = ( int * ) malloc( numberOfChunks * sizeof( int ) );

	if( index == NULL )
	{
		printf( "Failed to allocate memory in mergeInt" );
		exit( EXIT_FAILURE );
	}

	memset( index, 0, numberOfChunks * sizeof( int ) );

	for( i = 0; i < length; i++ )
	{
		min = unsorted[index[0]];
		minIndex = 0;

		int j;
		for( j = 0; j < numberOfChunks; j++ )
		{
			if( index[j] >= chunkSize )
				continue;

			if( unsorted[index[j] + j * chunkSize] < min )
			{
				min = unsorted[index[j] + j * chunkSize];
				minIndex = j;
			}
		} // end of for( j )

		sorted[i] = min;
		index[minIndex]++;
	} // end of for( i )

} // end of function mergeInt

void mergeLong( long * sorted, const long * unsorted, const int length, const int chunkSize )
{
	int numberOfChunks;
	int * index;
	long min;
	int minIndex;
	int i;

	numberOfChunks = ceil( ( double )length / chunkSize );

	index = ( int * ) malloc( numberOfChunks * sizeof( int ) );

	if( index == NULL )
	{
		printf( "Failed to allocate memory in mergeInt" );
		exit( EXIT_FAILURE );
	}

	memset( index, 0, numberOfChunks * sizeof( int ) );

	for( i = 0; i < length; i++ )
	{
		min = unsorted[index[0]];
		minIndex = 0;

		int j;
		for( j = 0; j < numberOfChunks; j++ )
		{
			if( index[j] >= chunkSize )
				continue;

			if( unsorted[index[j] + j * chunkSize] < min )
			{
				min = unsorted[index[j] + j * chunkSize];
				minIndex = j;
			}
		} // end of for( j )

		sorted[i] = min;
		index[minIndex]++;
	} // end of for( i )

} // end of function mergeLong

void mergeFloat( float * sorted, const float * unsorted, const int length, const int chunkSize )
{
	int numberOfChunks;
	int * index;
	float min;
	int minIndex;
	int i;

	numberOfChunks = ceil( ( double )length / chunkSize );

	index = ( int * ) malloc( numberOfChunks * sizeof( int ) );

	if( index == NULL )
	{
		printf( "Failed to allocate memory in mergeInt" );
		exit( EXIT_FAILURE );
	}

	memset( index, 0, numberOfChunks * sizeof( int ) );

	for( i = 0; i < length; i++ )
	{
		min = unsorted[index[0]];
		minIndex = 0;

		int j;
		for( j = 0; j < numberOfChunks; j++ )
		{
			if( index[j] >= chunkSize )
				continue;

			if( unsorted[index[j] + j * chunkSize] < min )
			{
				min = unsorted[index[j] + j * chunkSize];
				minIndex = j;
			}
		} // end of for( j )

		sorted[i] = min;
		index[minIndex]++;
	} // end of for( i )

} // end of function mergeFloat

void mergeDouble( double * sorted, const double * unsorted, const int length, const int chunkSize )
{
	int numberOfChunks;
	int * index;
	double min;
	int minIndex;
	int i;

	numberOfChunks = ceil( ( double )length / chunkSize );

	index = ( int * ) malloc( numberOfChunks * sizeof( int ) );

	if( index == NULL )
	{
		printf( "Failed to allocate memory in mergeInt" );
		exit( EXIT_FAILURE );
	}

	memset( index, 0, numberOfChunks * sizeof( int ) );

	for( i = 0; i < length; i++ )
	{
		min = unsorted[index[0]];
		minIndex = 0;

		int j;
		for( j = 0; j < numberOfChunks; j++ )
		{
			if( index[j] >= chunkSize )
				continue;

			if( unsorted[index[j] + j * chunkSize] < min )
			{
				min = unsorted[index[j] + j * chunkSize];
				minIndex = j;
			}
		} // end of for( j )

		sorted[i] = min;
		index[minIndex]++;
	} // end of for( i )

} // end of function mergeDouble

void merge( void * sorted, const void * unsorted, const int length, const int chunkSize, const char type )
{
	switch( type )
	{
	case 'I':
		mergeInt( ( int * ) sorted, ( int * ) unsorted, length, chunkSize );
		break;

	case 'L':
		mergeLong( ( long * ) sorted, ( long * ) unsorted, length, chunkSize );
		break;

	case 'F':
		mergeFloat( ( float * ) sorted, ( float * ) unsorted, length, chunkSize );
		break;

	case 'D':
		mergeDouble( ( double * ) sorted, ( double * ) unsorted, length, chunkSize );
		break;

	default:
		printf( "Cannot find the type of the data for merge process.\n" );
		exit( EXIT_FAILURE );
	} // end of switch
} // end of function merge

__global__ void bubbleSortKernelInt( int * ary, int length, int chunkSize, int numberOfChunks )
{
	int hold;
	int processThisChunk;
	int beginIndex;
	int endIndex;
	char isItAlreadySorted;

	processThisChunk = blockDim.x * blockIdx.x + threadIdx.x;

	if( processThisChunk >= numberOfChunks )
		return;

	beginIndex = processThisChunk * chunkSize;
	endIndex = beginIndex + chunkSize;

	if( endIndex == numberOfChunks - 1 )
		endIndex = length;

	endIndex--;

	isItAlreadySorted = 'N';
	int i = 0;

	while( isItAlreadySorted == 'N' )
	{
		isItAlreadySorted = 'Y';

		for( i = beginIndex; i < endIndex; i++ )
			if( ary[i] > ary[i+1] )
			{
				hold = ary[i];
				ary[i] = ary[i+1];
				ary[i+1] = hold;
				isItAlreadySorted = 'N';
			}
	} // end of while

} // end of kernel bubbleSortKernelInt

__global__ void bubbleSortKernelLong( long * ary, int length, int chunkSize, int numberOfChunks )
{
	long hold;
	int processThisChunk;
	int beginIndex;
	int endIndex;
	char isItAlreadySorted;

	processThisChunk = blockDim.x * blockIdx.x + threadIdx.x;

	if( processThisChunk >= numberOfChunks )
		return;

	beginIndex = processThisChunk * chunkSize;
	endIndex = beginIndex + chunkSize;

	if( endIndex == numberOfChunks - 1 )
		endIndex = length;

	endIndex--;

	isItAlreadySorted = 'N';
	int i = 0;

	while( isItAlreadySorted == 'N' )
	{
		isItAlreadySorted = 'Y';

		for( i = beginIndex; i < endIndex; i++ )
			if( ary[i] > ary[i+1] )
			{
				hold = ary[i];
				ary[i] = ary[i+1];
				ary[i+1] = hold;
				isItAlreadySorted = 'N';
			}
	} // end of while

} // end of kernel bubbleSortKernelLong

__global__ void bubbleSortKernelFloat( float * ary, int length, int chunkSize, int numberOfChunks )
{
	float hold;
	int processThisChunk;
	int beginIndex;
	int endIndex;
	char isItAlreadySorted;

	processThisChunk = blockDim.x * blockIdx.x + threadIdx.x;

	if( processThisChunk >= numberOfChunks )
		return;

	beginIndex = processThisChunk * chunkSize;
	endIndex = beginIndex + chunkSize;

	if( endIndex == numberOfChunks - 1 )
		endIndex = length;

	endIndex--;

	isItAlreadySorted = 'N';
	int i = 0;

	while( isItAlreadySorted == 'N' )
	{
		isItAlreadySorted = 'Y';

		for( i = beginIndex; i < endIndex; i++ )
			if( ary[i] > ary[i+1] )
			{
				hold = ary[i];
				ary[i] = ary[i+1];
				ary[i+1] = hold;
				isItAlreadySorted = 'N';
			}
	} // end of while

} // end of kernel bubbleSortKernelFloat

__global__ void bubbleSortKernelDouble( double * ary, int length, int chunkSize, int numberOfChunks )
{
	double hold;
	int processThisChunk;
	int beginIndex;
	int endIndex;
	char isItAlreadySorted;

	processThisChunk = blockDim.x * blockIdx.x + threadIdx.x;

	if( processThisChunk >= numberOfChunks )
		return;

	beginIndex = processThisChunk * chunkSize;
	endIndex = beginIndex + chunkSize;

	if( endIndex == numberOfChunks - 1 )
		endIndex = length;

	endIndex--;

	isItAlreadySorted = 'N';
	int i = 0;

	while( isItAlreadySorted == 'N' )
	{
		isItAlreadySorted = 'Y';

		for( i = beginIndex; i < endIndex; i++ )
			if( ary[i] > ary[i+1] )
			{
				hold = ary[i];
				ary[i] = ary[i+1];
				ary[i+1] = hold;
				isItAlreadySorted = 'N';
			}
	} // end of while

} // end of kernel bubbleSortKernelDouble



void launchGPU( void * ary, const int length, const int chunkSize, const int blockSize, const char type )
{
	void * gpuInput;
	size_t gpuArraySize;
	size_t elementSize;
	cudaError status;
	int numberOfChunks;

	dim3 block( blockSize, 1, 1 );
	dim3 grid( ( length/blockSize ) + 1, 1, 1 );

	numberOfChunks = ceil( length / chunkSize );

	switch( type )
	{
	case 'I':
		elementSize = sizeof( int );
		break;

	case 'L':
		elementSize = sizeof( long );
		break;

	case 'F':
		elementSize = sizeof( float );
		break;

	case 'D':
		elementSize = sizeof( double );
		break;

	default:
		printf( "Not sure what the input data type is.\n" );
		exit( EXIT_FAILURE );
	} // end of switch

	gpuArraySize = length * elementSize;

	status = cudaMalloc( ( void ** ) &gpuInput, gpuArraySize );

	if( status != cudaSuccess )
	{
		printf( "Failed to allocate memory for gpuInput.\n" );
		exit( EXIT_FAILURE );
	}

	status = cudaMemcpy( gpuInput, ary, gpuArraySize, cudaMemcpyHostToDevice );

	if( status != cudaSuccess )
	{
		printf( "Failed to copy content to gpuInput.\n" );
		exit( EXIT_FAILURE );
	}

	switch( type )
	{
	case 'I':
		bubbleSortKernelInt<<<grid, block>>>( ( int * ) gpuInput, length, chunkSize, numberOfChunks );
		break;

	case 'L':
		bubbleSortKernelLong<<<grid, block>>>( ( long * ) gpuInput, length, chunkSize, numberOfChunks );
		break;

	case 'F':
		bubbleSortKernelFloat<<<grid, block>>>( ( float * ) gpuInput, length, chunkSize, numberOfChunks );
		break;

	case 'D':
		bubbleSortKernelDouble<<<grid, block>>>( ( double * ) gpuInput, length, chunkSize, numberOfChunks );
		break;

	default:
		printf( "Unknown data type.\n" );
		exit( EXIT_FAILURE );
	}

	status = cudaMemcpy( ary, gpuInput, gpuArraySize, cudaMemcpyDeviceToHost );

	if( status != cudaSuccess )
	{
		printf( "Failed to copy content form gpuInput.\n" );
		exit( EXIT_FAILURE );
	}

} // end of function lauchGPU

void writeToFile( const double * ary, const int size )
{
	FILE * file;

	file = fopen( "sorted", "w" );

	if( file == NULL )
	{
		printf( "EEEEErrror.\n" );
		exit( 1 );
	}

	int i = 0;
	for( i = 0 ; i < size; i++ )
	{
		fprintf( file, "%f\n   ", ary[i]);
		if( ( i + 1) % 2 == 0 )
			printf("\n");

	}

	fclose( file );
}



int main( int argc, char ** argv )
{
	int howmany;
	char type;
	int chunkSize;
	int blockSize;
	void * unsorted;
	void * sorted;


	printf( "Working on it...\n");
	readCommandline( argc, argv, &howmany, &type, &chunkSize, &blockSize );
	unsorted = createArray( howmany, type );
	sorted = createArray( howmany, type );
	randomInitializer( unsorted, howmany, type );
	launchGPU( unsorted, howmany, chunkSize, blockSize, type );
	merge( sorted, unsorted, howmany, chunkSize, type );
	writeToFile(( double* )sorted, howmany);
	houseKeeping( sorted, unsorted );
	printf( "Done.\n");

} // end of function main
