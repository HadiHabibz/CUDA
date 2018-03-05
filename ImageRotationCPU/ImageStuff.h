struct ImgProp
{
	int Hpixels;
	int Vpixels;
	unsigned char HeaderInfo[54];
	unsigned long int Hbytes;
};

struct Pixel
{
	unsigned char R;
	unsigned char G;
	unsigned char B;
};

unsigned char** CreateBlankBMP(unsigned char FILL);
unsigned char* readBMP(char* );
void writeBMP( unsigned char * , const char * const );

extern struct ImgProp ip;


