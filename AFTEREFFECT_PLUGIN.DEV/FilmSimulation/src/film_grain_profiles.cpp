#ifndef FILM_PROFILES_H
#define FILM_PROFILES_H

#include "film_grain_profiles.hpp"

// Database of classic film presets mapped to procedural noise parameters
const std::vector<FilmProfile> FilmDatabase = 
{
    
    /* 
     * Film Stock: Kodak Vision3 500T (5219)
     * Introduced: 2007
     * Simulation Target: The modern Hollywood standard for night and interior scenes. 
     * Emulates large, chunky grain with dominant noise in the blue channel. 
     * The luma mask is biased toward shadows to create a soft, cinematic depth.
     */
    {
        "Kodak Vision3 500T (5219)", 
        "Modern Color Negative (Tungsten)",
        4, 
        0.85f, 1.0f, 0.55f, false, 
        0.35f, 1.2f, 0.9f,         
        false, false, false
    },

    /* 
     * Film Stock: Kodak Vision3 50D (5203)
     * Introduced: 2011
     * Simulation Target: Daylight-balanced film with microscopic, incredibly dense grain. 
     * Emulates tack-sharp details and a lack of chroma noise (RGB frequencies are nearly identical). 
     * Grain is strictly isolated to the midtones.
     */
    {
        "Kodak Vision3 50D (5203)", 
        "Modern Color Negative (Daylight)",
        2, 
        2.9f, 3.0f, 2.7f, false,   
        0.55f, 0.8f, 0.2f,         
        false, false, false
    },

    /* 
     * Film Stock: Kodak Vision3 250D (5207)
     * Introduced: 2009
     * Simulation Target: A highly versatile medium-speed film. Emulates a warm, organic texture. 
     * Features medium-sized grain with a broad luma response across the exposure range.
     */
    {
        "Kodak Vision3 250D (5207)", 
        "Modern Color Negative (Daylight)",
        3, 
        1.5f, 1.5f, 1.2f, false,   
        0.5f, 1.5f, 0.45f,         
        false, false, false
    },

    /* 
     * Film Stock: Eastman EXR 500T (5296)
     * Introduced: 1989
     * Simulation Target: The quintessential 90s blockbuster aesthetic (e.g., "Jurassic Park"). 
     * Emulates very large, "ragged" grain and introduces halation (glow around bright light sources) 
     * characteristic of the emulsion halation layers from that era.
     */
    {
        "Eastman EXR 500T (5296)", 
        "90s Color Negative (Tungsten)",
        4, 
        0.7f, 0.7f, 0.5f, false,   
        0.4f, 1.2f, 0.85f,         
        true, false, false         
    },

    /* 
     * Film Stock: Fujicolor Super F-500 (8572)
     * Introduced: 1990s (Discontinued ~2013)
     * Simulation Target: Kodak's main rival. Emulates a distinct "toxic" chroma footprint 
     * with larger grain in the green channel. Replicates Fuji's signature color shift, 
     * pulling deep shadows into a cyan/teal tint.
     */
    {
        "Fujicolor Super F-500 (8572)", 
        "Vintage Color Negative (Tungsten)",
        4, 
        0.9f, 0.6f, 0.8f, false,   
        0.45f, 1.1f, 0.8f,
        false, true, false         
    },

    /* 
     * Film Stock: Kodachrome 64
     * Introduced: 1935 (Peak popularity in the 70s-80s)
     * Simulation Target: The legendary color reversal (slide) film. Emulates tight grain 
     * and extreme contrast. The luma mask is exceptionally narrow, causing grain to drop off 
     * abruptly in both shadows and highlights, resulting in a "thick", punchy image.
     */
    {
        "Kodachrome 64", 
        "Color Reversal / Slide",
        2, 
        1.8f, 1.8f, 1.8f, false,   
        0.5f, 0.6f, 0.5f,          
        false, false, false
    },

    /* 
     * Film Stock: Kodak Tri-X 200 (5266)
     * Introduced: 1954
     * Simulation Target: The B&W classic. Emulates a harsh, monochromatic noise structure 
     * (isMonochrome = true; zero chroma noise). The luma mask is biased toward the highlights, 
     * keeping deep shadows clean for a deep, crushed "charcoal" black.
     */
    {
        "Kodachrome Tri-X 200 (5266)", 
        "Black & White Reversal",
        3, 
        1.0f, 1.0f, 1.0f, true,    
        0.6f, 1.0f, 0.7f,          
        false, false, false
    },

    /* 
     * Film Stock: Ilford HP5 Plus 400
     * Introduced: 1989 (Base emulsion dates back to the 1940s)
     * Simulation Target: Classic B&W photojournalism film. Emulates a large, gritty grain 
     * structure paired with a very wide luma mask. Grain survives deep into the shadows, 
     * creating a velvety, volumetric gray aesthetic.
     */
    {
        "Ilford HP5 Plus 400", 
        "Black & White Negative",
        4, 
        0.8f, 0.8f, 0.8f, true,    
        0.4f, 2.0f, 0.75f,         
        false, false, false
    },

    /* 
     * Film Stock: Technicolor Three-Strip
     * Introduced: 1932
     * Simulation Target: The Golden Age of Hollywood (e.g., "Gone with the Wind"). 
     * Emulates the dye-transfer process shot through three separate filters. Toggles the 
     * rgbShift flag to simulate microscopic spatial misalignment of the red, green, and blue 
     * emulsion layers (chromatic aberration / registration error).
     */
    {
        "Technicolor Three-Strip", 
        "Vintage Process",
        2, 
        1.5f, 1.5f, 1.5f, false,   
        0.5f, 1.0f, 0.6f,          
        false, false, true         
    },

    /* 
     * Film Stock: Kodak Vision3 200T (5213)
     * Introduced: 2010
     * Simulation Target: The go-to film for overcast days and twilight. Emulates a very tidy, 
     * tight, medium-sized grain. Overall intensity is rolled back to complement pastel, 
     * muted color palettes.
     */
    {
        "Kodak Vision3 200T (5213)", 
        "Modern Color Negative (Tungsten)",
        3, 
        1.8f, 2.0f, 1.5f, false,   
        0.5f, 1.0f, 0.35f,         
        false, false, false
    },

    /* 
     * Film Stock: ORWOcolor NC 21
     * Introduced: 1980s (GDR / East Germany)
     * Simulation Target: The signature "Eastern Bloc" cinematic look (e.g., DEFA studios). 
     * Emulates coarse, irregular grain clusters due to wider manufacturing tolerances compared to Kodak.
     * Features a pronounced chroma noise floor and triggers the cyan/magenta color shift typical 
     * of aging ORWO chemical dyes.
     */
    {
        "ORWOcolor NC 21", 
        "Vintage Color Negative (Eastern Bloc)",
        4, 
        0.75f, 0.85f, 0.65f, false,   
        0.45f, 1.4f, 0.75f,         
        false, true, false
    },

    /* 
     * Film Stock: Svema FN-64 (Свема ФН-64)
     * Introduced: 1980s-1990s (USSR)
     * Simulation Target: Classic Soviet black-and-white cinema (e.g., Tarkovsky's late era). 
     * Despite being a low-speed film (ISO 64), the emulsion coating inconsistencies result 
     * in a surprisingly chunky, highly irregular fractal grain structure. 
     * The luma mask is broad, keeping the midtones very gritty.
     */
    {
        "Svema FN-64", 
        "Black & White Negative (Soviet)",
        4, 
        1.1f, 1.1f, 1.1f, true,    
        0.5f, 1.5f, 0.65f,         
        false, false, false
    },

    /* 
     * Film Stock: Fomapan 400 Action
     * Introduced: 1990s (Czechoslovakia / Czech Republic)
     * Simulation Target: Gritty 90s documentary aesthetic. Emulates classic cubic silver halide 
     * crystals which appear much larger and harsher than modern T-Grain. 
     * Famous for lacking a robust anti-halation backing, thus the halation flag is forced to true 
     * to bloom the highlights.
     */
    {
        "Fomapan 400 Action", 
        "Black & White Negative (Czech)",
        3, 
        0.7f, 0.7f, 0.7f, true,    
        0.4f, 1.3f, 0.85f,         
        true, false, false
    }
};

#endif // FILM_PROFILES_H