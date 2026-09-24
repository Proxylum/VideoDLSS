#pragma once

#include <QColor>
#include <QIcon>
#include <QString>

class QApplication;
class QWidget;

namespace dlssvid {

// The look of dlssvid-gui after the mockups (Output/ux-mockups/*.dc.html, TASK-0017): the Fusion style with a dark
// palette (window and inputs #17181b, cards #1f2125, borders #2c2f35 / #3a3e46, text #e6e7e9, muted #a7abb3, accent
// #4aa3df with dark text #0b1116) and one stylesheet that fixes the sizes and paddings of the controls — secondary
// buttons 36 px high with 14 px side padding and a 6 px radius, the primary action in the accent colour, 24 px table
// buttons, 32 px segments in a bordered group, 30 px rounded chips, 30 px selects and inputs, cards padded 9×14 px.
// Variants are chosen with the `role` property (SetRole): "primary", "primary-big", "start", "start-primary", "small",
// "tall", "quiet" for buttons; "segment", "chip", "icon" for tool buttons; "segments" for a group frame; "card",
// "card-dashed" for frames; "muted", "hint", "title", "h1", "h2", "mono", "ok", "warn", "danger", "accent", "version",
// "cell-label" for labels. The start screen (TASK-0020) adds "section", "text-13", "hint-13", "dropzone-title",
// "dot-ok" / "dot-warn" / "dot-danger", "recent-title", "recent-meta", "preview" for labels and "recent" for a card.
void ApplyTheme(QApplication& app);
QString ThemeStyleSheet();
void SetRole(QWidget* widget, const char* role);  // sets the property and repolishes when the widget is already styled
// The mouse wheel over a combo box, spin box or slider that sits in a scrolling page scrolls the page instead of
// changing the value: a wheel over the project panel used to switch stage backends by accident. Controls outside a
// scroll area (the timeline's frame box) keep the wheel. Installed by ApplyTheme.
void InstallWheelGuard(QApplication& app);
// Playback glyphs drawn in the theme's text colour (Fusion's standard media icons are dark on the dark bar), and
// the drop zone's upload arrow; `color` overrides the ink (the drop zone draws it in the accent).
enum class Glyph { ToStart, Prev, Play, Pause, Next, Loop, Upload };
QIcon ThemeIcon(Glyph glyph, const QColor& color = QColor());
QColor ThemeAccent();

}  // namespace dlssvid
