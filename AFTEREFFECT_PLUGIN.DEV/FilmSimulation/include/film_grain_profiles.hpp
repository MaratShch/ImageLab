#ifndef __IMAGE_LAB_FILM_SIMULATION_POROFILES__
#define __IMAGE_LAB_FILM_SIMULATION_POROFILES__

#include <string>
#include <vector>

// Structure defining the mathematical parameters for film grain simulation
struct FilmProfile
{
    std::string name;          // Name of the film stock (e.g., "Kodak Vision3 500T (5219)")
    std::string type;          // Type description (e.g., "Color Negative, Modern")

                               // 1. Noise Generator Settings
    int octaves;               // Number of fBm octaves (1 to 4). Higher = clumpier texture.
    float freqR;               // Base frequency for the Red channel
    float freqG;               // Base frequency for the Green channel
    float freqB;               // Base frequency for the Blue channel (lower = larger grain)
    bool isMonochrome;         // If true, forces identical freqR/G/B and computes a single noise pass

                               // 2. Luma Masking Settings (D-Log E Curve Parabola)
    float lumaPeak;            // Where the grain is strongest (0.0 = absolute black, 1.0 = absolute white)
    float lumaWidth;           // Parabola width. Higher values cause grain to bleed further into deep shadows/highlights
    float intensity;           // Global blend opacity of the grain effect (0.0 to 1.0)

                               // 3. Extra Artifacts (Optional toggles for the render engine)
    bool halation;             // Emulates red/orange specular glow around bright highlights (e.g., EXR 500T)
    bool cyanShadowShift;      // Shifts deep black shadows towards a cyan/blue-green tint (e.g., Fuji F-500)
    bool rgbShift;             // Simulates physical 3-strip chromatic aberration / registration errors (Technicolor)
};

#endif // __IMAGE_LAB_FILM_SIMULATION_POROFILES__