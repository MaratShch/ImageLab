#ifndef ABSTRACTCOLORBARGENERATOR_H
#define ABSTRACTCOLORBARGENERATOR_H

#include <QColor>

class AbstractColorBarGenerator
{
public:
    virtual ~AbstractColorBarGenerator() = default;
    virtual QColor getColorAt(double position) const = 0; // position 0.0 (left) to 1.0 (right)
};

#endif // ABSTRACTCOLORBARGENERATOR_H