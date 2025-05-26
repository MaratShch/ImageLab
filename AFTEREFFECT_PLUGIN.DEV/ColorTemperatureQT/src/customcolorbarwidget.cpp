#include "customcolorbarwidget.h"
#include "abstractcolorbargenerator.h" // Include full definition
#include <QPainter>
#include <QPaintEvent>
#include <QDebug> // For potential warnings

CustomColorBarWidget::CustomColorBarWidget(const AbstractColorBarGenerator* generator, QWidget *parent)
    : QWidget(parent), m_colorGenerator(generator)
{
    if (!m_colorGenerator) {
        qWarning("CustomColorBarWidget: Color generator is null!");
        // In a real app, you might throw or set an error state
    }
    setMinimumSize(300, 30);
    // Example: Fixed size if you always want it this way
    // setFixedSize(500, 40);
}

CustomColorBarWidget::~CustomColorBarWidget()
{
    // This widget does not own m_colorGenerator
}

void CustomColorBarWidget::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    int w = width();
    int h = height();

    if (!m_colorGenerator) {
        painter.fillRect(rect(), Qt::darkGray);
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter, "No Color Generator");
        return;
    }

    for (int x = 0; x < w; ++x) {
        double position = (w > 1) ? static_cast<double>(x) / (w - 1) : 0.0;
        QColor color = m_colorGenerator->getColorAt(position);
        painter.setPen(color);
        painter.drawLine(x, 0, x, h);
    }
}

// Method to get the rendered content as a QImage
QImage CustomColorBarWidget::getRenderedImage() const
{
    // Create a QImage with the widget's current size and a suitable format
    // Format_ARGB32_Premultiplied is often good for rendering widgets
    QImage image(size(), QImage::Format_ARGB32_Premultiplied);
    image.fill(Qt::transparent); // Fill with transparent if you want alpha

    // Render the widget onto the QImage
    // this->render(&image); // This is a convenience function

    // OR, for more control or if render() has issues with specific scenarios,
    // you can manually create a QPainter and call paintEvent logic:
    QPainter painter(&image);
    // If your paintEvent relies on QPaintEvent details, you might need to simulate it.
    // For simple drawing like this, just painting directly is often fine.
    int w = image.width();
    int h = image.height();

    if (!m_colorGenerator) { // Redundant check if already handled, but safe
        painter.fillRect(image.rect(), Qt::darkGray);
        painter.setPen(Qt::white);
        painter.drawText(image.rect(), Qt::AlignCenter, "No Color Generator");
    } else {
        for (int x = 0; x < w; ++x) {
            double position = (w > 1) ? static_cast<double>(x) / (w - 1) : 0.0;
            QColor color = m_colorGenerator->getColorAt(position);
            painter.setPen(color);
            painter.drawLine(x, 0, x, h);
        }
    }
    // painter's destructor will finish painting

    return image;
}
