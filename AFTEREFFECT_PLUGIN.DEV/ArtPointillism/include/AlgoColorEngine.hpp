#include "CommonAuxPixFormat.hpp"


/**
 * Pre-process the target color based on Painter Mode.
 * UPDATED: Adds aggressive Chroma Expansion for Expressive modes to prevent "Gray Soup".
 */
inline fCIELabPix Apply_Color_Mode
(
    fCIELabPix input, 
    int color_mode,      // 0=Scientific, 1=Expressive
    float user_vibrancy  // -100 to +100 (from User UI)
) noexcept
{
    // 1. Calculate current Chroma (Saturation intensity)
    float chroma = std::sqrt(input.a * input.a + input.b * input.b);
    
    // 2. Base Boost from User Slider
    // Map -100..100 to factor 0.0..3.0
    float boost = 1.0f + (user_vibrancy / 50.0f);
    if (boost < 0.0f) boost = 0.0f;

    // 3. EXPRESSIVE MODE LOGIC (The Van Gogh Fix)
    if (color_mode == 1) // MODE_EXPRESSIVE
    {
        // A. Intrinsic Boost
        // Van Gogh is naturally more vibrant.
        boost *= 1.5f; 

        // B. The "Gray Killer" (Chroma Floor)
        // If a color is weak (grayish) but not pure black/white, 
        // we artificially inflate its color purity so it snaps to a colorful palette entry 
        // instead of a gray one.
        
        // Threshold: 5.0 is a subtle gray. 
        if (chroma < 10.0f && chroma > 0.5f) 
        {
            // If it's a weak color, pretend it's a strong color.
            // This forces the "Decompose" function to find a Blue/Yellow match 
            // instead of a Gray match.
            float fake_chroma = 20.0f + (user_vibrancy * 0.2f); 
            float scale = fake_chroma / chroma;
            
            input.a *= scale;
            input.b *= scale;
            
            // Recalculate chroma for the next step
            chroma = fake_chroma; 
        }
    }

    // 4. Apply Final Boost
    if (chroma > 0.001f && boost != 1.0f)
    {
        input.a *= boost;
        input.b *= boost;
    }

    return input;
}