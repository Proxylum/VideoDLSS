#include "DecodeText.h"

#include <QCoreApplication>

namespace dlssvid {

namespace {
QString Tr(const char* s) { return QCoreApplication::translate("DecodeText", s); }

QString LayoutText(const DecodeSupport& s) {  // «10 бит 4:4:4», «8 бит, монохромный»
    if (s.chroma == "420" || s.chroma == "422" || s.chroma == "444")
        return Tr("%1 бит %2:%3:%4").arg(s.bitDepth).arg(s.chroma[0]).arg(s.chroma[1]).arg(s.chroma[2]);
    return Tr("%1 бит, монохромный или особый").arg(s.bitDepth);
}
}  // namespace

// docs/ux-guidelines.md: what happened, why, what to do - in Russian; the core's English reason goes to the log only
QString SourceDecodeText(const DecodeSupport& s) {
    const QString gpu = QString::fromStdString(s.caps.gpu);
    const QString transcode = Tr("Перекодируйте ролик в H.264 или HEVC другой программой");
    switch (s.block) {
        case DecodeBlock::Layout:
            return Tr("Видео AV1 %1 не открыть: в этой версии программы AV1 декодирует только видеокарта, и только 8 и 10 бит 4:2:0. %2.")
                .arg(LayoutText(s), transcode);
        case DecodeBlock::NoCuda:
            return Tr("Видео AV1 не открыть: в этой версии программы AV1 декодирует только видеокарта NVIDIA, а она сейчас недоступна. Обновите "
                      "драйвер NVIDIA и перезапустите программу; подробности — в логе.");
        case DecodeBlock::NoNvdec:
            return Tr("Видео AV1 не открыть: в драйвере NVIDIA не нашёлся видеодекодер (NVDEC), а другого декодера AV1 в этой версии программы "
                      "нет. Переустановите драйвер NVIDIA и перезапустите программу.");
        case DecodeBlock::NoAv1:
            return Tr("Видео AV1 не открыть: видеокарта %1 не декодирует AV1 %2 бит, а другого декодера AV1 в этой версии программы нет. %3 "
                      "или откройте его на компьютере с видеокартой, которая декодирует AV1 (например, RTX 30 и новее).")
                .arg(gpu)
                .arg(s.bitDepth)
                .arg(transcode);
        case DecodeBlock::Size:
            return Tr("Видео AV1 не открыть: кадр %1×%2 вне пределов видеокарты %3 (от %4×%5 до %6×%7). Перекодируйте ролик другой программой "
                      "в H.264 или HEVC либо в кадр поменьше.")
                .arg(s.width)
                .arg(s.height)
                .arg(gpu)
                .arg(s.caps.minWidth)
                .arg(s.caps.minHeight)
                .arg(s.caps.maxWidth)
                .arg(s.caps.maxHeight);
        case DecodeBlock::NvdecFailed:
            return Tr("Видео AV1 не открыть: видеокарта отказалась его декодировать, а другого декодера AV1 в этой версии программы нет. Обновите "
                      "драйвер NVIDIA или %1.")
                .arg(transcode.left(1).toLower() + transcode.mid(1));
        case DecodeBlock::None: break;
    }
    return {};
}

QString SourceDecodeShort() { return Tr(" · не декодируется"); }

}  // namespace dlssvid
