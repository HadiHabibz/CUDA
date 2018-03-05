#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void readCommandline( int argc, char ** argv, int * howmany, char * type, int * chunkSize )
{
	if( argc != 4 )
	{
		printf( "Usage: %s <How many> <type> <chunk size> \n", argv[0] );
		exit( EXIT_SUCCESS );
	}

	*howmany = atoi( argv[1] );
	*howmany = *howmany * 1024 * 1024;

	*type = ( char ) toupper( ( char ) argv[2][0] );
	*chunkSize = atoi ( argv[3] );
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

void bubbleSortInt( int * ary, const int begin, const int end )
{
	char isItAlreadySorted;
	int i;

	isItAlreadySorted = 'N';

	while( isItAlreadySorted == 'N' )
	{
		isItAlreadySorted = 'Y';

		for( i = begin; i < end-1; i++ )
			if( ary[i] > ary[i+1] )
			{
				int hold;
				hold = ary[i];
				ary[i] = ary[i+1];
				ary[i+1] = hold;
				isItAlreadySorted = 'N';
			}

	} // end of while
} // end of function bubbleSortInt

void bubbleSortLong( long * ary, const int begin, const int end )
{
	char isItAlreadySorted;
	int i;

	isItAlreadySorted = 'N';

	while( isItAlreadySorted == 'N' )
	{
		isItAlreadySorted = 'Y';

		for( i = begin; i < end-1; i++ )
			if( ary[i] > ary[i+1] )
			{
				long hold;
				hold = ary[i];
				ary[i] = ary[i+1];
				ary[i+1] = hold;
				isItAlreadySorted = 'N';
			}

	} // end of while
} // end of function bubbleSortLong

void bubbleSortFloat( float * ary, const int begin, const int end )
{
	char isItAlreadySorted;
	int i;

	isItAlreadySorted = 'N';

	while( isItAlreadySorted == 'N' )
	{
		isItAlreadySorted = 'Y';

		for( i = begin; i < end-1; i++ )
			if( ary[i] > ary[i+1] )
			{
				float hold;
				hold = ary[i];
				ary[i] = ary[i+1];
				ary[i+1] = hold;
				isItAlreadySorted = 'N';
			}

	} // end of while
} // end of function bubbleSortFloat

void bubbleSortDouble( double * ary, const int begin, const int end )
{
	char isItAlreadySorted;
	int i;

	isItAlreadySorted = 'N';

	while( isItAlreadySorted == 'N' )
	{
		isItAlreadySorted = 'Y';

		for( i = begin; i < end-1; i++ )
			if( ary[i] > ary[i+1] )
			{
				double hold;
				hold = ary[i];
				ary[i] = ary[i+1];
				ary[i+1] = hold;
				isItAlreadySorted = 'N';
			}

	} // end of while
} // end of function bubbleSortDouble

void sortChunks( void * ary, const int length, const char type, const int chunkSize )
{
	int chunkCounter;
	int lastChunk;
	int endIndex;

	lastChunk = ceil( ( double )length / chunkSize );
	lastChunk--;

	for( chunkCounter = 0; chunkCounter <= lastChunk; chunkCounter++ )
	{
		endIndex = ( chunkCounter == lastChunk ) ? length : ( chunkCounter + 1 ) * chunkSize;

		switch( type )
		{
		case 'I':
			bubbleSortInt( ( int * ) ary, chunkCounter * chunkSize, endIndex );
			break;

		case 'L':
			bubbleSortLong( ( long * ) ary, chunkCounter * chunkSize, endIndex);
			break;

		case 'F':
			bubbleSortFloat( ( float * ) ary, chunkCounter * chunkSize, endIndex );
			break;

		case 'D':
			bubbleSortDouble( ( double * ) ary, chunkCounter * chunkSize, endIndex );
			break;

		default:
			printf( "Failed to sort the array: unknown data type.\n" );
			exit( EXIT_FAILURE );
		} // end of switch( type )

	} // end of for( chunkCounter )

} // end of function sortChunks

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
		fprintf( file, "%f\n   ", ary[i][0]);
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
	void * unsorted;
	void * sorted;

	printf( "Working on it \n" );

	readCommandline( argc, argv, &howmany, &type, &chunkSize );
	unsorted = createArray( howmany, type );
	sorted = createArray( howmany, type );
	randomInitializer( unsorted, howmany, type );
	sortChunks( unsorted, howmany, type, chunkSize );
	merge( sorted, unsorted, howmany, chunkSize, type );
	//writeToFile(sorted, howmany);
	houseKeeping( sorted, unsorted );

	printf( "Done!\n" );
	return 0;

} // end of function main
