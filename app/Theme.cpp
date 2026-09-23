#include "Theme.h"

#include <QAbstractScrollArea>
#include <algorithm>
#include <QAbstractSlider>
#include <QAbstractSpinBox>
#include <QApplication>
#include <QComboBox>
#include <QEvent>
#include <QFont>
#include <QPainter>
#include <QPainterPath>
#include <QPixmap>
#include <QScrollBar>
#include <QPalette>
#include <QStyle>
#include <QStyleFactory>
#include <QWidget>

namespace dlssvid {

namespace {

constexpr const char* kBg = "#17181b";
constexpr const char* kCard = "#1f2125";
constexpr const char* kCardDim = "#1a1c1f";
constexpr const char* kBorder = "#2c2f35";
constexpr const char* kControl = "#3a3e46";
constexpr const char* kText = "#e6e7e9";
constexpr const char* kMuted = "#a7abb3";
constexpr const char* kDisabled = "#6b7079";
constexpr const char* kAccent = "#4aa3df";
constexpr const char* kAccentHover = "#63b3e8";
constexpr const char* kAccentPressed = "#3c92cc";
constexpr const char* kOnAccent = "#0b1116";
constexpr const char* kAccentTint = "#1f3a4d";

class WheelGuard final : public QObject {
public:
    using QObject::QObject;
    bool eventFilter(QObject* watched, QEvent* event) override {
        if (event->type() != QEvent::Wheel) return false;
        auto* w = qobject_cast<QWidget*>(watched);
        if (!w) return false;
        const bool valueControl = qobject_cast<QComboBox*>(w) || qobject_cast<QAbstractSpinBox*>(w) || (qobject_cast<QAbstractSlider*>(w) && !qobject_cast<QScrollBar*>(w));
        if (!valueControl) return false;
        // only inside a scrolling page: there the wheel means "scroll", values change by click and keyboard (a focused
        // control is no exception — the pointer is usually left where the last click was); elsewhere (the timeline's
        // frame box) the wheel keeps changing the value
        bool inScrollArea = false;
        for (QWidget* p = w->parentWidget(); p && !inScrollArea; p = p->parentWidget()) inScrollArea = qobject_cast<QAbstractScrollArea*>(p) != nullptr;
        if (!inScrollArea) return false;
        if (QWidget* parent = w->parentWidget()) QApplication::sendEvent(parent, event);  // the scroll area above gets the wheel
        return true;
    }
};

}  // namespace

QString ThemeStyleSheet() {
    QString css = R"(
QMainWindow::separator { background: @border; width: 1px; height: 1px; }
QMenuBar { background: @bg; border-bottom: 1px solid @border; padding: 2px 8px; }
QMenuBar::item { padding: 6px 10px; border-radius: 4px; color: @muted; background: transparent; }
QMenuBar::item:selected { background: @card; color: @text; }
QMenu { background: @card; border: 1px solid @control; padding: 4px; }
QMenu::item { padding: 6px 28px 6px 12px; border-radius: 4px; }
QMenu::item:selected { background: @tint; }
QMenu::item:disabled { color: @disabled; }
QMenu::separator { height: 1px; background: @border; margin: 4px 8px; }
QStatusBar { background: @card; border-top: 1px solid @border; color: @muted; font-size: 12px; min-height: 28px; }
QStatusBar::item { border: 0; }
QDockWidget { color: @muted; font-size: 12px; }
QDockWidget::title { background: @card; padding: 6px 10px; border-bottom: 1px solid @border; text-align: left; }
QTabBar::tab { background: @bg; color: @muted; padding: 6px 12px; border: 1px solid @border; border-bottom: 0; }
QTabBar::tab:selected { background: @card; color: @text; }
QScrollArea { border: 0; }
QScrollBar:vertical { background: @bg; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: @control; border-radius: 5px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: @bg; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: @control; border-radius: 5px; min-width: 24px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QToolTip { background: @card; color: @text; border: 1px solid @control; padding: 4px 8px; }

QPushButton { min-height: 34px; padding: 0 14px; border-radius: 6px; border: 1px solid @control; background: @card; color: @text; }
QPushButton:hover { border-color: @accent; }
QPushButton:pressed { background: @bg; }
QPushButton:disabled { color: @disabled; border-color: @border; background: @dim; }
QPushButton:flat { background: transparent; border: 1px solid transparent; min-height: 22px; padding: 0 9px; border-radius: 4px; font-size: 12px; color: @muted; }
QPushButton:flat:hover { border-color: @control; color: @text; }
QPushButton[role="small"] { min-height: 22px; padding: 0 9px; border-radius: 4px; font-size: 12px; background: @bg; }
QPushButton[role="tall"] { min-height: 42px; padding: 0 12px; }
QPushButton[role="start"] { min-height: 44px; padding: 0 22px; font-size: 15px; font-weight: 500; }
QPushButton[role="quiet"] { min-height: 44px; padding: 0 12px; background: @border; color: @muted; border: 0; }
QPushButton[role="primary"], QPushButton[role="primary-big"], QPushButton[role="start-primary"] {
    background: @accent; color: @on-accent; font-weight: 600; border: 0; min-height: 36px; padding: 0 16px; }
QPushButton[role="primary-big"] { min-height: 48px; font-size: 16px; }
QPushButton[role="start-primary"] { min-height: 46px; padding: 0 22px; font-size: 15px; }
QPushButton[role="primary"]:hover, QPushButton[role="primary-big"]:hover, QPushButton[role="start-primary"]:hover { background: @accent-hover; }
QPushButton[role="primary"]:pressed, QPushButton[role="primary-big"]:pressed, QPushButton[role="start-primary"]:pressed { background: @accent-pressed; }
QPushButton[role="primary"]:disabled, QPushButton[role="primary-big"]:disabled, QPushButton[role="start-primary"]:disabled { background: @border; color: @disabled; }

QToolButton { min-height: 26px; padding: 0 8px; border-radius: 4px; border: 1px solid transparent; background: transparent; color: @text; }
QToolButton:hover { background: @card; border-color: @control; }
QToolButton:checked { background: @tint; border-color: @accent; }
QToolButton:disabled { color: @disabled; }
QToolButton[role="icon"] { min-height: 28px; min-width: 28px; padding: 0 6px; }
QToolButton[role="segment"] { min-height: 32px; padding: 0 10px; border: 0; border-radius: 0; background: @bg; color: @muted; }
QToolButton[role="segment"]:hover { background: @card; color: @text; }
QToolButton[role="segment"]:checked { background: @accent; color: @on-accent; font-weight: 600; }
QFrame[role="segments"] { border: 1px solid @control; border-radius: 4px; background: @bg; }
QToolButton[role="chip"] { min-height: 28px; padding: 0 12px; border-radius: 14px; border: 1px solid @control; background: @bg; color: @text; }
QToolButton[role="chip"]:hover { border-color: @accent; }
QToolButton[role="chip"]:checked { border-color: @accent; background: @tint; }
QToolButton[role="chip"][popupMode="1"] { padding-right: 22px; }
QToolButton::menu-button { border: 0; width: 18px; }

QComboBox { min-height: 28px; padding: 0 8px; border-radius: 4px; border: 1px solid @control; background: @bg; color: @text; }
QComboBox:hover { border-color: @accent; }
QComboBox:disabled { color: @disabled; border-color: @border; }
QComboBox::drop-down { border: 0; width: 22px; subcontrol-origin: padding; subcontrol-position: center right; }
QComboBox QAbstractItemView { background: @card; border: 1px solid @control; selection-background-color: @tint; outline: 0; padding: 4px; }
QSpinBox, QDoubleSpinBox, QLineEdit { min-height: 28px; padding: 0 8px; border-radius: 4px; border: 1px solid @control; background: @bg; color: @text; }
QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover { border-color: @accent; }
QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled { color: @disabled; border-color: @border; }
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 16px; border: 0; background: transparent; }
QPlainTextEdit, QTextEdit { border: 1px solid @border; border-radius: 4px; background: @bg; padding: 4px 6px; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid @control; border-radius: 3px; background: @bg; }
QCheckBox::indicator:hover { border-color: @accent; }
QCheckBox::indicator:checked { background: @accent; border-color: @accent; }
QCheckBox::indicator:disabled { border-color: @border; }
QSlider::groove:horizontal { height: 4px; background: @border; border-radius: 2px; }
QSlider::sub-page:horizontal { background: @accent; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: @text; border: 2px solid @accent; }
QProgressBar { background: @border; border: 0; border-radius: 4px; text-align: center; color: @text; font-size: 12px; }
QProgressBar::chunk { background: @accent; border-radius: 4px; }

QTableWidget, QTableView, QListWidget, QTreeView { background: @bg; alternate-background-color: @dim; border: 1px solid @border; gridline-color: @border; outline: 0; }
QTableView::item, QListWidget::item { padding: 4px 8px; }
QTableView::item:selected, QListWidget::item:selected { background: @tint; color: @text; }
QHeaderView::section { background: @card; color: @muted; padding: 4px 8px; border: 0; border-bottom: 1px solid @border; border-right: 1px solid @border; font-size: 12px; }

QLabel[role="muted"] { color: @muted; }
QLabel[role="hint"] { color: @muted; font-size: 12px; }
QLabel[role="title"] { font-size: 14px; font-weight: 600; }
QLabel[role="h1"] { font-size: 30px; font-weight: 600; }
QLabel[role="h2"] { font-size: 20px; font-weight: 600; }
QLabel[role="lead"] { font-size: 15px; color: @muted; }
QLabel[role="mono"] { font-family: "Consolas", "Cascadia Mono", monospace; }
QLabel[role="ok"] { color: #5fbf7a; }
QLabel[role="warn"] { color: #e0a84f; }
QLabel[role="danger"] { color: #ff7a7a; }
QLabel[role="accent"] { color: @accent; }
QLabel[role="version"] { font-family: "Consolas", "Cascadia Mono", monospace; font-size: 11px; color: @muted; border: 1px solid @control; border-radius: 4px; padding: 1px 5px; }
QLabel[role="cell-label"] { padding: 4px 9px; border-radius: 4px; background: @card; font-size: 12px; font-weight: 600; }
QFrame[role="card"] { background: @card; border: 1px solid @border; border-radius: 6px; }
QFrame[role="card-dashed"] { background: @dim; border: 1px dashed @border; border-radius: 6px; }
QFrame[role="dropzone"] { border: 1.5px dashed @accent; border-radius: 10px; background: #1a1d21; }
QFrame[role="dropzone"]:hover { background: @card; }
QWidget[role="bar"] { border-bottom: 1px solid @border; }
)";
    css.replace("@accent-hover", kAccentHover)
        .replace("@accent-pressed", kAccentPressed)
        .replace("@on-accent", kOnAccent)
        .replace("@accent", kAccent)
        .replace("@tint", kAccentTint)
        .replace("@border", kBorder)
        .replace("@control", kControl)
        .replace("@card", kCard)
        .replace("@dim", kCardDim)
        .replace("@bg", kBg)
        .replace("@text", kText)
        .replace("@muted", kMuted)
        .replace("@disabled", kDisabled);
    return css;
}

void ApplyTheme(QApplication& app) {
    app.setStyle(QStyleFactory::create("Fusion"));
    QPalette p;
    const QColor bg(kBg), card(kCard), text(kText), muted(kMuted), disabled(kDisabled), accent(kAccent), tint(kAccentTint), border(kBorder);
    p.setColor(QPalette::Window, bg);
    p.setColor(QPalette::WindowText, text);
    p.setColor(QPalette::Base, bg);
    p.setColor(QPalette::AlternateBase, QColor(kCardDim));
    p.setColor(QPalette::Text, text);
    p.setColor(QPalette::PlaceholderText, disabled);
    p.setColor(QPalette::Button, card);
    p.setColor(QPalette::ButtonText, text);
    p.setColor(QPalette::BrightText, text);
    p.setColor(QPalette::Highlight, tint);
    p.setColor(QPalette::HighlightedText, text);
    p.setColor(QPalette::Link, accent);
    p.setColor(QPalette::ToolTipBase, card);
    p.setColor(QPalette::ToolTipText, text);
    p.setColor(QPalette::Light, QColor(kControl));
    p.setColor(QPalette::Midlight, border);
    p.setColor(QPalette::Mid, border);
    p.setColor(QPalette::Dark, bg);
    p.setColor(QPalette::Shadow, bg);
    for (QPalette::ColorRole role : {QPalette::WindowText, QPalette::Text, QPalette::ButtonText}) p.setColor(QPalette::Disabled, role, disabled);
    p.setColor(QPalette::Disabled, QPalette::Highlight, border);
    app.setPalette(p);
    QFont font("Segoe UI", 10);  // 13 px at 96 dpi, the mockups' base size
    app.setFont(font);
    app.setStyleSheet(ThemeStyleSheet());
    InstallWheelGuard(app);
}

void InstallWheelGuard(QApplication& app) {
    static WheelGuard* guard = nullptr;
    if (guard) return;
    guard = new WheelGuard(&app);
    app.installEventFilter(guard);
}

QIcon ThemeIcon(Glyph glyph) {
    QIcon icon;
    for (int size : {16, 20, 24, 32, 48}) {
        QPixmap pm(size, size);
        pm.fill(Qt::transparent);
        QPainter p(&pm);
        p.setRenderHint(QPainter::Antialiasing);
        const QColor ink(kText);
        p.setPen(Qt::NoPen);
        p.setBrush(ink);
        const qreal s = size;
        auto triangle = [&](qreal x0, qreal x1, bool right) {  // a play-style triangle between x0 and x1 (fractions of the size)
            QPainterPath path;
            const qreal top = s * 0.22, bottom = s * 0.78, mid = s * 0.5;
            if (right) {
                path.moveTo(s * x0, top);
                path.lineTo(s * x1, mid);
                path.lineTo(s * x0, bottom);
            } else {
                path.moveTo(s * x1, top);
                path.lineTo(s * x0, mid);
                path.lineTo(s * x1, bottom);
            }
            path.closeSubpath();
            p.drawPath(path);
        };
        auto bar = [&](qreal x) { p.drawRect(QRectF(s * x, s * 0.22, std::max(1.5, s * 0.1), s * 0.56)); };
        switch (glyph) {
            case Glyph::ToStart: bar(0.2); triangle(0.36, 0.8, false); break;
            case Glyph::Prev: triangle(0.28, 0.72, false); break;
            case Glyph::Play: triangle(0.3, 0.78, true); break;
            case Glyph::Pause: bar(0.28); bar(0.6); break;
            case Glyph::Next: triangle(0.28, 0.72, true); break;
            case Glyph::Loop: {  // a circular arrow: an open ring with an arrowhead at its gap
                QPen pen(ink, std::max(1.5, s * 0.1));
                pen.setCapStyle(Qt::RoundCap);
                p.setPen(pen);
                p.setBrush(Qt::NoBrush);
                const QRectF ring(s * 0.2, s * 0.2, s * 0.6, s * 0.6);
                p.drawArc(ring, 30 * 16, 300 * 16);
                p.setPen(Qt::NoPen);
                p.setBrush(ink);
                QPainterPath head;  // at the arc's end (30°): pointing along the ring
                const qreal cx = s * 0.5 + s * 0.3 * 0.866, cy = s * 0.5 - s * 0.3 * 0.5;
                head.moveTo(cx + s * 0.02, cy - s * 0.14);
                head.lineTo(cx + s * 0.12, cy + s * 0.06);
                head.lineTo(cx - s * 0.12, cy + s * 0.02);
                head.closeSubpath();
                p.drawPath(head);
                break;
            }
        }
        p.end();
        icon.addPixmap(pm);
    }
    return icon;
}

void SetRole(QWidget* widget, const char* role) {
    widget->setProperty("role", role);
    if (widget->isVisible()) {  // already polished: apply the new selector
        widget->style()->unpolish(widget);
        widget->style()->polish(widget);
    }
}

}  // namespace dlssvid
