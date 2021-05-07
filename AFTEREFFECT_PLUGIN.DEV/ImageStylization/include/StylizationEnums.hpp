#pragma once

typedef enum
{
	eSTYLE_NONE = 0,
	eSTYLE_NEWS_PAPER_OLD,
	eSTYLE_NEWS_PAPER_COLOR,
	eSTYLE_GLASSY_EFFECT,
	eSTYLE_OIL_PAINT,
	eSTYLE_CARTOON,
	eSTYLE_SKETCH_PENCIL,
	eSTYLE_SKETCH_CHARCOAL,
	eSTYLE_IMPRESSIONISM,
	eSTYLE_TOTAL_EFFECTS
}eSTYLIZATION;


constexpr static char strStyleEffect[] =
{
	"None|"
	"Old News Paper|"
	"Color News Paper|"
	"Glassy Effect|"
	"Oil Paint Image|"
	"Cartoon|"
	"Pencil Sketch|"
	"Charcoal Sketch|"
	"Impressionism"
};

constexpr int32_t toatal_variants = 2;
constexpr int32_t param_name_length = PF_MAX_EFFECT_PARAM_NAME_LEN + 1;

/* glass effect min/max settinig */
constexpr int32_t glassyMin = 0;
constexpr int32_t glassyMax = 40;
constexpr int32_t glassyDefault = glassyMin;

constexpr char StyleSlider1[toatal_variants][param_name_length] =
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