#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define BMP_HEADER_SIZE 54
#define THREADCOUNT 9
#define MAXTHREADS 127


struct ImgProp
{
	int Hpixels;
	int Vpixels;
	unsigned char HeaderInfo[BMP_HEADER_SIZE];
	unsigned long int Hbytes;
};

struct ImgProp ip;
struct ImgProp ip2;
unsigned char ** image;
unsigned char ** shrinkedImage;
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

void checkCommandlineInput( int argc, char ** argv )
{
	if( argc != 5 )
	{
		printf( "Usage: %s <input image> <output image> <X> <Y>\n", argv[0] );
		exit( EXIT_FAILURE );
	}

} // end of function checkCommandlineInput

unsigned char ** readBMP( char* filename )
{
	unsigned char tmp;
	unsigned char **TheImage;

	FILE * file = fopen( filename, "rb" );

	if( file == NULL )
	{
		printf( "\n\n%s NOT FOUND\n\n", filename );
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

	TheImage = ( unsigned char ** )malloc( height * sizeof(unsigned char*) );

	i = 0;
	for(; i<height; i++ )
		TheImage[i] = ( unsigned char * )malloc( RowBytes * sizeof(unsigned char) );

	for(i = 0; i < height; i++)
		fread( TheImage[i], sizeof(unsigned char), RowBytes, file );

	fclose( file );

	return TheImage;
} // end of function readBMP

void writeBMP( unsigned char ** img, char* filename )
{
	FILE * file = fopen( filename, "wb" );

	if( file == NULL )
	{
		printf( "\n\nFILE CREATION ERROR: %s\n\n",filename );
		exit( EXIT_FAILURE );
	}

	unsigned long int x,y;
	unsigned char temp;

	ip.Hpixels /= xFactor;
	ip.Vpixels /= yFactor;
	ip.Hbytes = ( ip.Hpixels*3 + 3 ) & ( ~3 );

	* ( ( int * ) &ip.HeaderInfo[2] ) = (ip.HbytesR * ip.VpixelsR +  BMP_HEADER_SIZE );
	* ( ( int * ) &ip.HeaderInfo[18] ) = ip.Hpixels;
	* ( ( int * ) &ip.HeaderInfo[22] ) = ip.Vpixels;

	//write header
	for( x = 0; x < BMP_HEADER_SIZE; x++ )
		fputc( ip.HeaderInfo[x], file );

	//write data
	for( x=0; x<ip.Vpixels; x++ )
		for( y=0; y < ip.Hbytes; y++ )
		{
			temp=img[x][y];
			fputc( temp, file );
		}

	printf( "\n  Output BMP File name: %20s  (%u x %u)\n", filename, ip.Hpixels, ip.Vpixels );

	fclose( file );
} // end of function writeBMP

void * shrinkXY( void * myID )
{
	int id;
	int chunk;
	int beginIndex;
	int endIndex;
	int currentRow;
	int currentCol;
	int temp;
	int i;
	int j;

	id = * ( int * ) myID;
	chunk = ip.Vpixels / THREADCOUNT;
	beginIndex = id * chunk;
	endIndex = beginIndex + chunk;

	if( id == THREADCOUNT - 1 )
		endIndex = ip.Vpixels;

	currentRow = ( ip.Vpixels / ( yFactor * THREADCOUNT ) ) * id ;
	currentCol = -1;

	while( beginIndex % yFactor != 0 )
		beginIndex++;

	for( i = beginIndex; i < endIndex; i += yFactor )
	{
		for( j = 0; j < ip.Hpixels; j += xFactor )
		{
			currentCol++;
			temp = j * 3;
			shrinkedImage[currentRow][currentCol++] = image[i][temp];
			shrinkedImage[currentRow][currentCol++] = image[i][temp+1];
			shrinkedImage[currentRow][currentCol] = image[i][temp+2];
		}

		currentCol = -1;
		currentRow++;
	}

	return 0;

} // end of function shrinkY

void threadLauncher( void )
{
	int i;
	pthread_t threadHandle[MAXTHREADS];
	pthread_attr_t threadAttribute;
	int threadId[MAXTHREADS];
	int hBytes;

	pthread_attr_init( &threadAttribute );
	pthread_attr_setdetachstate( &threadAttribute, PTHREAD_CREATE_JOINABLE );
	hBytes = ( ( ip.Hbytes / xFactor ) ) + 3 & ( ~3 );

	shrinkedImage = ( unsigned char ** ) malloc( ( ( ip.Vpixels / yFactor ) + 1 ) * sizeof( unsigned char * ) );

	for( i = 0; i < (ip.Vpixels / yFactor) + 1; i++ )
		shrinkedImage[i] = ( unsigned char * ) malloc( hBytes * sizeof( unsigned char ) );

	startTime = getTime();
	for( i = 0; i < THREADCOUNT; i++ )
	{
		threadId[i] = i;

		int status;

		status = pthread_create( &threadHandle[i], &threadAttribute, shrinkXY, ( void * ) &threadId[i] );

		if( status != 0 )
		{
			printf( "Cannot create thread %d.\n", i );
			exit( EXIT_FAILURE );
		}
	}

	for( i = 0; i < THREADCOUNT; i++ )
		pthread_join( threadHandle[i], NULL );

	endTime = getTime();

	pthread_attr_destroy( &threadAttribute );

} // end of function threadLauncher


int main( int argc, char ** argv )
{
	checkCommandlineInput( argc, argv );
	xFactor = atoi( argv[3] );
	yFactor = atoi( argv[4] );

	image = readBMP( argv[1] );
	threadLauncher();
	writeBMP( shrinkedImage, argv[2] );
	printf( "GPU Execution Time: %0.2f ms\n\n", (endTime - startTime) / 1.0 );

	return 0;
} // end of function main
