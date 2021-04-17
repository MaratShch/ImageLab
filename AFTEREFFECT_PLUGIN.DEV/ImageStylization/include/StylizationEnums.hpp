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

static constexpr char strStylePopup[] = "Image Style";