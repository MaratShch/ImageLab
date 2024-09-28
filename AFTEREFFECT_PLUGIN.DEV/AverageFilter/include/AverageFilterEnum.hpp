#ifndef __IMAGE_LAB_AVERAGE_FILTER_ENUMERATORS__
#define __IMAGE_LAB_AVERAGE_FILTER_ENUMERATORS__

typedef enum {
	eAEVRAGE_FILTER_INPUT,
	eAEVRAGE_FILTER_WINDOW_SIZE,
	eAVERAGE_FILTER_GEOMETRIC_AVERAGE,
	eAVERAGE_FILTER_TOTAL_PARAMETERS
}eAVERAGE_FILTER_ITEMS;

typedef enum {
	eAVERAGE_WINDOW_3x3,
	eAVERAGE_WINDOW_5x5,
	eAVERAGE_WINDOW_7x7,
	eAVERAGE_WINDOW_TOTAL
}eAVERAGE_FILTER_WINDOW_SZIE;

constexpr char FilterWindowSizesStr[]  = { "3 x 3|" "5 x 5|" "7 x 7" };
constexpr char FilterWindowSizeStr[]   = "Window Size";
constexpr char GeomethricCheckBoxStr[] = "Geometric Average";

#endif
