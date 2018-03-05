#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

#define error1 1
#define error2 2
#define error3 3
#define error4 4
#define error5 5
#define error6 6
#define error7 7 

#define programIterations 5
#define maxStringSize 256
#define maxNumberOfThreads 128
#define numberOfAllNumbers 8192
#define maxNumberOfDigits 25
#define lastInput -99

// This struct holds the summary of user's command
struct Commandline
{
	unsigned char numberOfThreads;
	unsigned char optimizationLevel;
	char inputFilename[maxStringSize];
	char outputFilename[maxStringSize];
}; // end of struct CommandLine

struct ThreadArguments
{
	int threadID;
	long int * numberArray;
	long int * primeArray;
	int * indexArray;
}; // end of struct ThreadArguments

struct Commandline commandline;
void * ( * threadOperation ) ( void * );

// read the input and output file names and save them in commandline 
// structure, also check for restrictions and errors
void extractPath( const char * source, char * destination )
{
	if( strlen( source ) > 256 )
	{
		printf( "Error%d: %s is too long. it should have less than 256 characters.\n", error2, source );
		exit( error2 );
	}

	strcpy( destination, source );
} // end of function extractPath

// Read the commandline and determine what the number of threads
// is, check for error and ambiguities
void extractNumberOfThreads( const char * numberOfThreads )
{
	int number;

	// In case the number of threads is not entered
	// select the default value
	if( numberOfThreads[0] == '-' )
		return;

	if( numberOfThreads[0] < '1' || numberOfThreads[0] > '9' )
	{
		printf( "Error%d: Number of threads should begin with a digit other than zero.\n", error3,  numberOfThreads );
		exit( error3 );
	}	

	number = atoi( numberOfThreads );

	if( number >= 128 || number <= 0 )
	{
		printf( "Warning: %d is invalid for number of threads. It is replaced by 1.\nMay be the ", number );
		printf( "input number is too large (should be below 128). It also cannot be 0 or negative.\n" );
		commandline.numberOfThreads = 1;
		return;
	}

	commandline.numberOfThreads = number;
		
} // end of function extractNumberOfThreads

void extractOptimizationLevel( const char * option )
{
	if( option[0] >= '1' && option[0] <= '9' )
		return;

	else if( strcmp( option, "-optimization=0" ) == 0 )
		commandline.optimizationLevel = 0;

	else if( strcmp( option, "-optimization=1" ) == 0 )
		commandline.optimizationLevel = 1;

	else if( strcmp( option, "-optimization=2" ) == 0 )
		commandline.optimizationLevel = 2;
	
	else	 
	{
		printf( "Error%d: %s in not a valid option.\n", error4, option );
		exit( error4 );
	} 
} // end of function extractOptimizationLevel

int readCommandline( int argc, char ** argv )
{
	// Set the default value
	commandline.optimizationLevel = 0;
	commandline.numberOfThreads = 1;

	if( argc < 3 || argc > 5 )
	{
		printf( "Error%d: please use the following format:\n", error1 );
		printf( "%s inputFile.txt outputFile.txt [optional: [1-127]] [optional:-optimization=[0-3]]\n", argv[0] );
		exit( error1 );
	}

	extractPath( argv[1], commandline.inputFilename );
	extractPath( argv[2], commandline.outputFilename );

	if( argc >= 4 )
	{
		extractNumberOfThreads( argv[3] );
		extractOptimizationLevel( argv[3] );
	}

	if( argc == 5 )
	{ 
		extractNumberOfThreads( argv[4] );
		extractOptimizationLevel( argv[4] );
	}


} // end of function readCommandline

// Read the input file and save the numbers in an array
void readFile( long int * array, FILE * source )
{
	char holder[maxStringSize];
	
	for( int i = 0; fgets( holder, maxNumberOfDigits, source ) != NULL; i++ )
		array[i] =  atol( holder );

} // end of function readFile

// Check if the input is a prime number or not
char isPrime( long int number )
{
	long int squareRoot;

	squareRoot = (long int ) sqrt( ( double ) number );

	if( number <= 1 )
		return 'n';

	if( number == 2 )
		return 'y';

	// Check if the number is even
	if ( ( number & 1 ) == 0 ) 
		return 'n';

	for( unsigned long int i = 3; i <= squareRoot; i += 2 )
		if( number % i == 0 )
			return 'n';

	return 'y';

} // end of function isPrime_singleThread

// Find all the prime numbers in the input arrya
// Save the prime numbers in outputArray
// Save the location of the prime numbers in indexArray
void findPrimeNumbers_singleThread( long int * inputArray, long int * outputArray, int * indexArray )
{
	for( int i = 0; i < numberOfAllNumbers; i++ )
		if( isPrime( inputArray[i] ) == 'y' )
		{
			outputArray[i] = inputArray[i];
			indexArray[i] = i+1;
		}
} // end of function findPrimeNumber

// Write the content of numberArray on the destination file
// in this format: location in source file : number
void writeFile( long int * numberArray, int * indexArray, FILE * destination )
{
	for( int i = 0; i < numberOfAllNumbers; i++)
		if( indexArray[i] != 0 )
			fprintf( destination, "%d : %ld\n", indexArray[i], numberArray[i] );
} // end of function writeFile

// return the system time in micro seconds
double getCurrentTimeInus( )
{
	struct timeval now;

	gettimeofday( &now, NULL );

	return ( ( double ) now.tv_sec * 1000000 + ( ( double ) now.tv_usec ) );
} // end of function getCurrentTimeInus

// Each thread searches a portion of the file to identify the prime numbers
void * findPrimeNumbers_multiThread_1( void * data )
{
	struct ThreadArguments *  threadArguments;
	int chunk;
	int beginIndex;
	int finishIndex;

	// Use these three variables to eliminate extra reference
	long int * primeArray;
	long int * numberArray;
	int * indexArray;

	threadArguments = ( struct ThreadArguments * ) data;
	primeArray = threadArguments->primeArray;
	numberArray = threadArguments->numberArray;
	indexArray = threadArguments->indexArray;

	chunk = numberOfAllNumbers / commandline.numberOfThreads;
	beginIndex = chunk * threadArguments->threadID;
	finishIndex = beginIndex + chunk;

	if( threadArguments->threadID == commandline.numberOfThreads-1 )
		finishIndex = numberOfAllNumbers;

	for( int i = beginIndex; i < finishIndex; i++ )
		if( isPrime( numberArray[i] ) == 'y' )
		{
			primeArray[i] = numberArray[i];
			indexArray[i] = i+1; 
		}
} // end of function findPrimeNumbers_multiThread_0



// Each thread searches a portion of the file to identify the prime numbers
void * findPrimeNumbers_multiThread_0( void * data )
{
	struct ThreadArguments *  threadArguments;
	int chunk;
	int beginIndex;
	int finishIndex;

	threadArguments = ( struct ThreadArguments * ) data;
	chunk = numberOfAllNumbers / commandline.numberOfThreads;
	beginIndex = chunk * threadArguments->threadID;
	finishIndex = beginIndex + chunk;

	if( threadArguments->threadID == commandline.numberOfThreads-1 )
		finishIndex = numberOfAllNumbers;

	for( int i = beginIndex; i < finishIndex; i++ )
		if( isPrime( threadArguments->numberArray[i] ) == 'y' )
		{
			threadArguments->primeArray[i] = threadArguments->numberArray[i];
			threadArguments->indexArray[i] = i+1; 
		}
} // end of function findPrimeNumbers_multiThread_0


// Create and run threads
void executeMultiThread( long int * allNumbers, long int * primeNumbers, int * indexArray )
{
	struct ThreadArguments threadArguments[maxNumberOfThreads];
	pthread_t threadHandle[maxNumberOfThreads];
	pthread_attr_t threadAttribute;
	int threadCreationError;

	pthread_attr_init( &threadAttribute );
	pthread_attr_setdetachstate( &threadAttribute, PTHREAD_CREATE_JOINABLE );

	if( commandline.optimizationLevel == 0 )
		threadOperation = &findPrimeNumbers_multiThread_0;

	else if( commandline.optimizationLevel == 1 )
		threadOperation = &findPrimeNumbers_multiThread_1;

	for( int i = 0; i < commandline.numberOfThreads; i++ )
	{
		threadArguments[i].threadID = i;
		threadArguments[i].numberArray = allNumbers;
		threadArguments[i].primeArray = primeNumbers;
		threadArguments[i].indexArray = indexArray;

		threadCreationError = pthread_create( &threadHandle[i], &threadAttribute, threadOperation, ( void * ) &threadArguments[i] );		

		if( threadCreationError != 0 )
		{
			printf( "Error%d: Cannot create thread %d.\n", error7, i );
			exit( error7 );
		}
	}
	
	for( int i = 0; i < commandline.numberOfThreads; i++ )
		pthread_join( threadHandle[i], NULL );
}

// Based on user's options, pick the right function and execute the program
void runTheProgram( long int * allNumbers, long int * primeNumbers, int * indexArray )
{
	if( commandline.numberOfThreads == 1 )
		findPrimeNumbers_singleThread( allNumbers, primeNumbers, indexArray );

	if( commandline.numberOfThreads > 1 )
		executeMultiThread( allNumbers, primeNumbers, indexArray );

} // end of function runTheProgram

int main( int argc, char ** argv  )
{
	FILE * sourceFile;
	FILE * destinationFile;
	long int allNumbers[numberOfAllNumbers] = {0};
	long int primeNumbers[numberOfAllNumbers] = {0};
	int indexArray[numberOfAllNumbers] = {0};
	double startTime;
	double finishTime;

	readCommandline( argc, argv );
	sourceFile = fopen( commandline.inputFilename, "r" );
	
	if( sourceFile == NULL )
	{
		printf( "Error%d: cannot open %s file.\n", error5, commandline.inputFilename );
		exit( error5 );
	}	

	readFile( allNumbers, sourceFile );

	startTime = getCurrentTimeInus();

	for( int i = 0; i < programIterations; i++ )
	{
		printf( "\rProcessing... %d%%", ( int ) ( i * 100 / programIterations ) ) ;
		fflush( stdout );
		runTheProgram( allNumbers, primeNumbers, indexArray );
	}

	printf( "\rProcessing...100%%\n" ) ;

	finishTime = getCurrentTimeInus();

	destinationFile = fopen( commandline.outputFilename, "w" );
	
	if( destinationFile == NULL )
	{
		printf( "Error%d: cannot open %s file.\n", error6, commandline.outputFilename );
		exit( error6 );
	}

	writeFile( primeNumbers, indexArray, destinationFile );

	fclose( sourceFile );
	fclose( destinationFile );	

	printf( "Total execution time: %.0f ms\n", ( finishTime - startTime ) / 1000 );
	printf( "Average execution time: %.0f ms\n", ( finishTime - startTime ) / ( 1000 * programIterations ) );
	
} // end of function main
