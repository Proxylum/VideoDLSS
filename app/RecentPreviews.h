#pragma once

#include <QHash>
#include <QImage>
#include <QObject>
#include <QSet>
#include <QString>
#include <QThreadPool>
#include <optional>

namespace dlssvid {

// Previews for the start screen's recent cards (mockup «1 · Стартовый экран»): a 96×40 frame of the video and its
// «1920×800 24 fps» line — with the result's «→ 3840×1600 48 fps» and the result's frame when the result exists.
// Made once per file on a worker thread (FFmpeg on the CPU: the first frame that is not black) and cached as a PNG
// under the cache directory, keyed by the files' paths, sizes and modification times, so the start screen never
// waits for a decoder and a re-encoded result gets a fresh preview.
class RecentPreviews : public QObject {
    Q_OBJECT
public:
    struct Preview {
        QImage image;  // kWidth×kHeight at kScale (cover-cropped); null when the video could not be decoded
        QString meta;  // «1920×800 24 fps» / «1920×800 24 fps → 3840×1600 48 fps»; empty when the file could not be opened
    };
    static constexpr int kWidth = 96, kHeight = 40, kScale = 2;

    explicit RecentPreviews(QObject* parent = nullptr);
    ~RecentPreviews() override;

    // The preview of `video` (and `result`): from memory, else nullopt and a worker makes it — previewReady(video)
    // follows on this object's thread.
    std::optional<Preview> get(const QString& video, const QString& result = {});
    // Makes one synchronously (the worker; tests): the disk cache first, then the decoder.
    static Preview Make(const QString& video, const QString& result = {});
    static QString CacheDir();
    static void SetCacheDir(const QString& dir);  // tests: away from the user's cache
    static QString CacheFile(const QString& video, const QString& result);  // where Make() keeps the PNG

signals:
    void previewReady(const QString& video);

private:
    QHash<QString, Preview> ready_;
    QSet<QString> pending_;
    QThreadPool pool_;
};

}  // namespace dlssvid
