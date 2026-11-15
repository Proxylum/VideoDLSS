#include "DecodeText.h"

#include <QCoreApplication>

namespace dlssvid {

namespace {
QString Tr(const char* s) { return QCoreApplication::translate("DecodeText", s); }
}  // namespace

QString SourceDecodeText(const DecodeSupport& s) {
    const QString gpu = QString::fromStdString(s.caps.gpu);
    switch (s.block) {
        case DecodeBlock::Layout:
            return Tr("Видео не открыть: AV1 в формате %1. В этой сборке AV1 декодирует только видеокарта NVIDIA (NVDEC), а она берёт AV1 лишь 8 и "
                      "10 бит 4:2:0.")
                .arg(QString::fromStdString(s.pixelFormat));
        case DecodeBlock::NoCuda:
            return Tr("Видео не открыть: AV1 в этой сборке декодирует только видеокарта NVIDIA (NVDEC), а CUDA недоступна (%1).")
                .arg(QString::fromStdString(s.caps.why));
        case DecodeBlock::NoAv1:
            return Tr("Видео не открыть: видеокарта %1 не декодирует AV1 %2 бит, а программного декодера AV1 в этой сборке нет. AV1 декодируют, "
                      "например, RTX 30 и новее.")
                .arg(gpu)
                .arg(s.bitDepth);
        case DecodeBlock::Size:
            return Tr("Видео не открыть: кадр AV1 %1×%2 вне пределов видеокарты %3 (от %4×%5 до %6×%7).")
                .arg(s.width)
                .arg(s.height)
                .arg(gpu)
                .arg(s.caps.minWidth)
                .arg(s.caps.minHeight)
                .arg(s.caps.maxWidth)
                .arg(s.caps.maxHeight);
        case DecodeBlock::NvdecFailed:
            return Tr("Видео не открыть: видеокарта не начала декодировать этот AV1, а программного декодера AV1 в этой сборке нет. Подробности — "
                      "в журнале.");
        case DecodeBlock::None: break;
    }
    return {};
}

QString SourceDecodeShort() { return Tr(" · не декодируется"); }

}  // namespace dlssvid
