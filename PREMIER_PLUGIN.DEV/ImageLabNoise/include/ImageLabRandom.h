#pragma once

#define ROTL(d,lrot) ((d<<(lrot)) | (d>>(8*sizeof(d)-(lrot))))

static inline unsigned int utils_get_random_value(void)
{
	// used xorshift algorithm
	static unsigned int x = 123456789u;
	static unsigned int y = 362436069u;
	static unsigned int z = 521288629u;
	static unsigned int w = 88675123u;
	static unsigned int t;

	t = x ^ (x << 11);
	x = y; y = z; z = w;
	return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
}

static unsigned int xState = utils_get_random_value();
static unsigned int yState = utils_get_random_value();
static unsigned int zState = utils_get_random_value();


inline unsigned int romuTrio32_random (void)
{
	unsigned int xp = xState, yp = yState, zp = zState;
	xState = 3323815723u * zp;
	yState = yp - xp; yState = ROTL(yState, 6);
	zState = zp - yp; zState = ROTL(zState, 22);
	return xp;
}

