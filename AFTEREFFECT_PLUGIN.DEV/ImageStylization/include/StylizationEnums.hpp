#pragma once
#include <cstdint>
#include "AE_Effect.h"

using SequenceIdT = uint32_t;
using GaussianT   = float;

typedef enum
{
	eSTYLE_NONE = 0,
	eSTYLE_NEWS_PAPER_OLD,
	eSTYLE_NEWS_PAPER_COLOR,
	eSTYLE_GLASSY_EFFECT,
	eSTYLE_CARTOON,
	eSTYLE_SKETCH_PENCIL,
	eSTYLE_SKETCH_CHARCOAL,
	eSTYLE_PAINT,
	eSTYLE_OIL_PAINT,
	eSTYLE_IMPRESSIONISM,
	eSTYLE_POINTILLISM,
	eSTYLE_MOSAIC,
	eSTYLE_CUBISM,
	eSTYLE_TOTAL_EFFECTS
}eSTYLIZATION;


constexpr static char strStyleEffect[] =
{
	"None|"
	"Old News Paper|"
	"Color News Paper|"
	"Glassy Effect|"
	"Cartoon|"
	"Sketch: Pencil|"
	"Sketch: Charcoal|"
	"Art: Paint|"
	"Art: Oil Paint|"
	"Art: Impressionism|"
	"Art: Pointillism|"
	"Art: Mosaic|"
	"Art: Cubism"
};

constexpr int32_t total_variants = 2;
constexpr int32_t param_name_length = PF_MAX_EFFECT_PARAM_NAME_LEN + 1;

/* glass effect min/max settinig */
constexpr int32_t glassyMin = 0;
constexpr int32_t glassyMax = 40;
constexpr int32_t glassyDefault = glassyMin;

constexpr char StyleSlider1[total_variants][param_name_length] =
{
	"N/A",
	"Dispersion"
};


typedef enum {
	IMAGE_STYLE_INPUT,
	IMAGE_STYLE_POPUP,
	IMAGE_STYLE_SLIDER1,
	IMAGE_STYLE_TOTAL_PARAMS
}Item;


static constexpr char strStylePopup[] = "Image Style";