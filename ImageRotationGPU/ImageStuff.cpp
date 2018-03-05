#include "ImageStuff.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

struct ImgProp ip;

unsigned char** CreateBlankBMP(unsigned char FILL)
{
    int i,j;

	unsigned char** img = (unsigned char **)malloc(ip.Vpixels * sizeof(unsigned char*));
    for(i=0; i<ip.Vpixels; i++){
        img[i] = (unsigned char *)malloc(ip.Hbytes * sizeof(unsigned char));
		memset((void *)img[i],FILL,(size_t)ip.Hbytes); // zero out every pixel
    }
    return img;
}


unsigned char * readBMP( char * filename )
{
	int i;
	unsigned char HeaderInfo[54];
	unsigned char *theImage;

	FILE * f;

	f = fopen( filename, "rb" );

	if( f == NULL )
	{
		printf( "%s NOT FOUND\n",filename );
		exit( EXIT_FAILURE );
	}

	// read the 54-byte header
	fread( HeaderInfo, sizeof( unsigned char ), 54, f);

	// extract image height and width from header
	int width = * ( int * )&HeaderInfo[18];
	int height = * ( int * )&HeaderInfo[22];

	//copy header for re-use
	for( i = 0; i < 54; i++ )
		ip.HeaderInfo[i] = HeaderInfo[i];

	ip.Vpixels = height;
	ip.Hpixels = width;
	ip.Hbytes = ( width * 3 + 3 ) & ( ~3 );

	printf( "Input BMP File name: %20s (%u x %u)\n", filename, ip.Hpixels, ip.Vpixels );

	unsigned char tmp;

	theImage = ( unsigned char * )malloc( ip.Hbytes * ip.Vpixels * sizeof( unsigned char ) );
	fread( theImage, sizeof( unsigned char ), ip.Hbytes * ip.Vpixels, f );
	fclose(f);

	// remember to free() it in caller!
	return theImage;
} // end of function readBMP


void writeBMP( unsigned char * img, const char * const filename )
{

	unsigned long int x,y;
	unsigned long int size;

	FILE * f = fopen( filename, "wb" );

	if( f == NULL )
	{
		printf( "FILE CREATION ERROR: %s\n", filename );
		exit( EXIT_FAILURE );
	}

	size = ip.Vpixels * ip.Hbytes;

	//write header
	for( x = 0; x < 54; x++)
		fputc( ip.HeaderInfo[x], f );

	//write data
	for( x = 0; x < size; x++ )
		fputc( img[x], f );

	printf( "Output BMP File name: %20s  (%u x %u)\n", filename, ip.Hpixels, ip.Vpixels );
	fclose( f );
} // end of function writeBMP
