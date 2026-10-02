// ---------------------------------------------------------------------------
// ADDENDUM for Algo2Rgb2XYZ.hpp — ITU-R BT.2020 primaries, D65 white.
// Paste into Algo2Rgb2XYZ.hpp, then compile with
// -DIMAGELAB2_HAVE_REC2020_MATRICES to enable workingSpace == 1.
//
// Same derivation as the sRGB pair: exact primaries/white in 50-digit
// arithmetic, literals are the correctly-rounded double of the exact values,
// and the inverse comes from the SAME exact forward — never from a
// separately-rounded copy. Row sums equal D65 XYZ: RGB(1,1,1) -> the white
// point (0.95045593, 1, 1.08905775), identical to the sRGB pair.
// Checked: M * M^-1 - I is 3.5e-17.
// ---------------------------------------------------------------------------

CACHE_ALIGN constexpr double Rec2020toXYZ_f64[9] =
{
    0.636958048301291324, 0.144616903586208378, 0.168880975164172054,
    0.262700212011267031, 0.677998071518871037, 0.059301716469861945,
    0.000000000000000000, 0.028072693049087508, 1.060985057710790880,
};

CACHE_ALIGN constexpr double XYZtoRec2020_f64[9] =
{
     1.716651187971267590, -0.355670783776392385, -0.253366281373659796,
    -0.666684351832488975,  1.616481236634939030,  0.015768545813911131,
     0.017639857445310915, -0.042770613257808655,  0.942103121235474017,
};
