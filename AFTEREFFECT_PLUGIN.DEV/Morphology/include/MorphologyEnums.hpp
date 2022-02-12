#pragma once

typedef enum {
	MORPHOLOGY_FILTER_INPUT,
	MORPHOLOGY_OPERATION_TYPE,
	MORPHOLOGY_ELEMENT_TYPE,
	MORPHOLOGY_KERNEL_SIZE,
	MORPHOLOGY_FILTER_TOTAL_PARAMS
}Item;

typedef enum {
	SE_OP_NONE,
	SE_OP_EROSION,
	SE_OP_DILATION,
	SE_OP_OPEN,
	SE_OP_CLOSE,
	SE_OP_THIN,
	SE_OP_THICK,
	SE_OP_GRADIENT,
	SE_OP_TOTAL
}SeOperation;

typedef enum {
	SE_TYPE_SQUARE,
	SE_TYPE_VERTICAL,
	SE_TYPE_HORIZONTAL,
	SE_TYPE_CROSS,
	SE_TYPE_FRAME,
	SE_TYPE_RING,
	SE_TYPE_DISK,
	SE_TYPE_DIAMOND,
	SE_TYPE_TOTALS
}SeType;

typedef enum {
	SE_SIZE_3x3,
	SE_SIZE_5x5,
	SE_SIZE_7x7,
	SE_SIZE_9x9,
	SE_SIZE_TOTALS
}SeSize;