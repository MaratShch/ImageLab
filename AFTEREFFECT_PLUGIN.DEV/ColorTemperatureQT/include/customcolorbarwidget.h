#ifndef CUSTOMCOLORBARWIDGET_H
#define CUSTOMCOLORBARWIDGET_H

#include <QWidget>
#include <QImage> // For rendering to an image

class AbstractColorBarGenerator; // Forward declaration

class CustomColorBarWidget : public QWidget
{
    Q_OBJECT

public:
    explicit CustomColorBarWidget(const AbstractColorBarGenerator* generator, QWidget *parent = nullptr);
    ~CustomColorBarWidget();

    // Method to get the rendered content as a QImage
    QImage getRenderedImage() const;

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    const AbstractColorBarGenerator* m_colorGenerator;
};

#endif // CUSTOMCOLORBARWIDGET_H