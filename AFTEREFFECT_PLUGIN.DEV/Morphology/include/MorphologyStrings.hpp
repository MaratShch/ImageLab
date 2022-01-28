#pragma once

static constexpr char MorphOperationType[]{ "Morphology Type" };
static constexpr char MorphSeType[]{ "Structured Element" };
static constexpr char MorphSeSize[]{ "Element Size" };

constexpr char strMorphOperation[] =
{
	"None|"
	"Erosion|"
	"Dilation|"
	"Open|"
	"Close|"
	"Thin|"
	"Thick|"
	"Gradient"
};

constexpr char strStructuredElement[] =
{
	"Square|"
	"Vertical|"
	"Horizontal|"
	"Cross|"
	"Frame|"
	"Ring|"
	"Disk|"
	"Diamond"
};

constexpr char strElemSize[] =
{
	"3 x 3|"
	"5 x 5|"
	"7 x 7|"
	"9 x 9"
};

