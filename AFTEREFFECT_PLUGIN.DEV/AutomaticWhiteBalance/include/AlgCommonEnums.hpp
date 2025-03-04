#ifndef __IMAGE_LAB_AUTHOMATIC_WB_ALGO_ENUMS__
#define __IMAGE_LAB_AUTHOMATIC_WB_ALGO_ENUMS__

typedef enum
{
    DAYLIGHT = 0,
    OLD_DAYLIGHT,
    OLD_DIRECT_SUNLIGHT_AT_NOON,
    MID_MORNING_DAYLIGHT,
    NORTH_SKY_DAYLIGHT,
    DAYLIGHT_FLUORESCENT_F1,
    COOL_FLUERESCENT,
    WHITE_FLUORESCENT,
    WARM_WHITE_FLUORESCENT,
    DAYLIGHT_FLUORESCENT_F5,
    COOL_WHITE_FLUORESCENT,
    TOTAL_ILLUMINATES
}eILLUMINATE;

typedef enum
{
    CHROMATIC_CAT02 = 0,
    CHROMATIC_VON_KRIES,
    CHROMATIC_BRADFORD,
    CHROMATIC_SHARP,
    CHROMATIC_CMCCAT2000,
    TOTAL_CHROMATIC
}eChromaticAdaptation;

#endif // __IMAGE_LAB_AUTHOMATIC_WB_ALGO_ENUMS__