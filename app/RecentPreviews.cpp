#include "RecentPreviews.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFileInfo>
#include <QMetaObject>
#include <QPainter>
#include <QStandardPaths>
#include <cmath>
#include <filesystem>

#include "convert/ColorConvert.h"
#include "io/VideoDecoder.h"
#include "passes/PassImage.h"
#include "util/Log.h"

namespace dlssvid {

namespace {

QString g_cacheDir;

std::filesystem::path FsPath(const QString& p) { return std::filesystem::path(p.toStdWString()); }

QString FpsText(const Rational& r) {
    const double v = r.ToDouble();
    if (v <= 0.0) return "?";
    if (std::abs(v - std::round(v)) < 1e-3) return QString::number(static_cast<int>(std::lround(v)));
    return QString::number(v, 'f', 2);
}

QString SizeLine(const VideoStreamInfo& i) { return QString("%1×%2 %3 fps").arg(i.width).arg(i.height).arg(FpsText(i.frameRate)); }

bool Probe(const QString& file, VideoStreamInfo& out) {
    if (file.isEmpty()) return false;
    try {
        VideoDecoder dec(FsPath(file), VideoDecoder::Probe());
        out = dec.Info();
        return out.width > 0 && out.height > 0;
    } catch (const std::exception& e) {
        Log()->debug("recent preview: cannot probe '{}': {}", file.toStdString(), e.what());
        return false;
    }
}

// The first frame that is not a black lead-in (at most a dozen decoded), as an 8-bit RGB QImage.
QImage FirstFrame(const QString& file) {
    QImage last;
    try {
        VideoDecoder dec(FsPath(file));  // CPU decode: one frame, any GPU stays free for the pipeline
        const ColorInfo color = ColorInfoFromStream(dec.Info());
        CpuFrame f;
        for (int i = 0; i < 12 && dec.NextFrame(f); ++i) {
            const PassImage rgb = Yuv420pToRgb(f, color, PixelType::U8);
            if (rgb.width == 0 || rgb.ChannelCount() != 3) break;
            last = QImage(rgb.data.data(), static_cast<int>(rgb.width), static_cast<int>(rgb.height), static_cast<qsizetype>(rgb.RowBytes()),
                          QImage::Format_RGB888)
                       .copy();
            // mean luma of a coarse grid: skip fades from black
            double sum = 0.0;
            int n = 0;
            for (int y = 0; y < last.height(); y += std::max(1, last.height() / 16))
                for (int x = 0; x < last.width(); x += std::max(1, last.width() / 16)) {
                    sum += qGray(last.pixel(x, y));
                    ++n;
                }
            if (n > 0 && sum / n > 16.0) break;
        }
    } catch (const std::exception& e) {
        Log()->debug("recent preview: cannot decode '{}': {}", file.toStdString(), e.what());
    }
    return last;
}

// object-fit: cover — scaled to fill w×h, the overflow cropped around the centre.
QImage Cover(const QImage& src, int w, int h) {
    const QImage scaled = src.scaled(w, h, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);
    const int x = std::max(0, (scaled.width() - w) / 2), y = std::max(0, (scaled.height() - h) / 2);
    return scaled.copy(x, y, w, h);
}

QString Stamp(const QString& file) {
    const QFileInfo fi(file);
    return file + "|" + QString::number(fi.size()) + "|" + QString::number(fi.lastModified().toMSecsSinceEpoch());
}

}  // namespace

RecentPreviews::RecentPreviews(QObject* parent) : QObject(parent) { pool_.setMaxThreadCount(1); }

RecentPreviews::~RecentPreviews() { pool_.waitForDone(); }

QString RecentPreviews::CacheDir() {
    if (!g_cacheDir.isEmpty()) return g_cacheDir;
    return QStandardPaths::writableLocation(QStandardPaths::CacheLocation) + "/recent-previews";
}

void RecentPreviews::SetCacheDir(const QString& dir) { g_cacheDir = dir; }

QString RecentPreviews::CacheFile(const QString& video, const QString& result) {
    const QByteArray key = QCryptographicHash::hash((Stamp(video) + "\n" + (result.isEmpty() ? QString() : Stamp(result))).toUtf8(), QCryptographicHash::Sha1);
    return CacheDir() + "/" + QString::fromLatin1(key.toHex()) + ".png";
}

RecentPreviews::Preview RecentPreviews::Make(const QString& video, const QString& result) {
    const QString file = CacheFile(video, result);
    Preview p;
    if (QImage cached; cached.load(file)) {
        p.image = cached;
        p.meta = cached.text("meta");
        return p;
    }
    VideoStreamInfo src, res;
    if (Probe(video, src)) p.meta = SizeLine(src);
    if (Probe(result, res)) p.meta += (p.meta.isEmpty() ? "" : " → ") + SizeLine(res);
    const QImage frame = FirstFrame(result.isEmpty() ? video : result);
    if (!frame.isNull()) p.image = Cover(frame, kWidth * kScale, kHeight * kScale);
    if (!p.image.isNull()) {  // the cache holds decoded previews only: an unreadable file is retried next time
        QDir().mkpath(CacheDir());
        QImage out = p.image;
        out.setText("meta", p.meta);
        out.save(file, "PNG");
    }
    return p;
}

std::optional<RecentPreviews::Preview> RecentPreviews::get(const QString& video, const QString& result) {
    const QString key = video + "\n" + result;
    if (const auto it = ready_.constFind(key); it != ready_.constEnd()) return *it;
    if (!pending_.contains(key)) {
        pending_.insert(key);
        pool_.start([this, video, result, key] {
            Preview p = Make(video, result);
            // delivered on this object's thread; dropped if the object is gone by then
            QMetaObject::invokeMethod(
                this,
                [this, key, video, p = std::move(p)]() mutable {
                    ready_.insert(key, std::move(p));
                    pending_.remove(key);
                    emit previewReady(video);
                },
                Qt::QueuedConnection);
        });
    }
    return std::nullopt;
}

}  // namespace dlssvid
