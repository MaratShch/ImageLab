#ifndef MYCCTCOLORGENERATOR_H
#define MYCCTCOLORGENERATOR_H

#include "abstractcolorbargenerator.h"
#include <QColor>
#include <algorithm> // For std::max and std::min

class MyCCTColorGenerator : public AbstractColorBarGenerator
{
public:
    QColor getColorAt(double position) const override
    {
        // ** YOUR CCT to RGB LOGIC GOES HERE **
        // This is a placeholder for reddish -> white -> bluish.
        position = std::max(0.0, std::min(position, 1.0)); // Clamp position to 0.0-1.0

        int r, g, b;
        const double midPoint = 0.5; // White point

        if (position < midPoint) {
            // From Red (255,0,0) to White (255,255,255)
            double t = position / midPoint; // Normalize 0-1 for this segment
            r = 255;
            g = static_cast<int>(t * 255.0);
            b = static_cast<int>(t * 255.0);
        } else {
            // From White (255,255,255) to Blue (0,0,255)
            double t = (position - midPoint) / (1.0 - midPoint); // Normalize 0-1 for this segment
            r = static_cast<int>((1.0 - t) * 255.0);
            g = static_cast<int>((1.0 - t) * 255.0);
            b = 255;
        }
        return QColor(std::max(0, std::min(255, r)),
                      std::max(0, std::min(255, g)),
                      std::max(0, std::min(255, b)));
    }
};

#endif // MYCCTCOLORGENERATOR_H