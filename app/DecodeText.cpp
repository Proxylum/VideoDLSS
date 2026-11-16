#include "DecodeText.h"

#include <QCoreApplication>

namespace dlssvid {

namespace {
QString Tr(const char* s) { return QCoreApplication::translate("DecodeText", s); }

QString LayoutText(const DecodeSupport& s) {  // «10 бит 4:4:4», «8 бит, монохромный или особый формат»
    if (s.chroma == "420" || s.chroma == "422" || s.chroma == "444")
        return Tr("%1 бит %2:%3:%4").arg(s.bitDepth).arg(s.chroma[0]).arg(s.chroma[1]).arg(s.chroma[2]);
    return Tr("%1 бит, монохромный или особый формат").arg(s.bitDepth);
}
}  // namespace

// docs/ux-guidelines.md: what happened, why, what to do - in Russian; the core's English reason goes to the log only. «не декодируется на
// этом компьютере», not «не открыть»: the project is open, the video is listed and may even be shown from its scrubbing proxy
QString SourceDecodeText(const DecodeSupport& s) {
    const QString gpu = QString::fromStdString(s.caps.gpu);
    const QString transcode = Tr("перекодируйте ролик в H.264 или HEVC другой программой");
    switch (s.block) {
        case DecodeBlock::Layout:
            return Tr("Видео AV1 не декодируется на этом компьютере: его формат (%1) эта версия программы не декодирует — AV1 декодирует только "
                      "видеокарта, и только 8 и 10 бит 4:2:0. Что сделать: %2.")
                .arg(LayoutText(s), transcode);
        case DecodeBlock::NoCuda:
            return Tr("Видео AV1 не декодируется на этом компьютере: в этой версии программы AV1 декодирует только видеокарта NVIDIA, а она сейчас "
                      "недоступна для программы. Что сделать: проверьте, что видеокарта NVIDIA включена и драйвер NVIDIA установлен, и перезапустите "
                      "программу или %1; подробности — в логе.")
                .arg(transcode);
        case DecodeBlock::NoNvdec:
            return Tr("Видео AV1 не декодируется на этом компьютере: в драйвере NVIDIA не нашёлся видеодекодер (NVDEC), а другого декодера AV1 в этой "
                      "версии программы нет. Что сделать: переустановите драйвер NVIDIA и перезапустите программу или %1.")
                .arg(transcode);
        case DecodeBlock::NoAv1:
            return Tr("Видео AV1 не декодируется на этом компьютере: видеокарта %1 не декодирует AV1 %2 бит, а другого декодера AV1 в этой версии "
                      "программы нет. Что сделать: %3 или откройте его на компьютере с видеокартой, которая декодирует AV1 (например, RTX 30 и новее).")
                .arg(gpu)
                .arg(s.bitDepth)
                .arg(transcode);
        case DecodeBlock::Size: {
            const bool small = s.width < s.caps.minWidth || s.height < s.caps.minHeight;
            return Tr("Видео AV1 не декодируется на этом компьютере: кадр %1×%2 %3, чем декодирует видеокарта %4 (от %5×%6 до %7×%8). Что сделать: %9.")
                .arg(s.width)
                .arg(s.height)
                .arg(small ? Tr("меньше") : Tr("больше"))
                .arg(gpu)
                .arg(s.caps.minWidth)
                .arg(s.caps.minHeight)
                .arg(s.caps.maxWidth)
                .arg(s.caps.maxHeight)
                .arg(small ? Tr("перекодируйте ролик в H.264 или HEVC другой программой") : Tr("перекодируйте ролик другой программой в H.264 или HEVC "
                                                                                              "либо в кадр поменьше"));
        }
        case DecodeBlock::NvdecFailed:
            return Tr("Видео AV1 не декодируется на этом компьютере: видеокарта отказалась его декодировать, а другого декодера AV1 в этой версии "
                      "программы нет. Что сделать: обновите драйвер NVIDIA и перезапустите программу или %1.")
                .arg(transcode);
        case DecodeBlock::NvdecStartFailed:
            return Tr("Видеокарта сейчас не смогла начать декодировать этот AV1 — скорее всего, не хватило её памяти; кадры загрузятся снова при "
                      "следующей перемотке или после конца обработки. Что сделать: дождитесь "
                      "конца обработки или закройте программы, которые занимают видеокарту.");
        case DecodeBlock::None: break;
    }
    return {};
}

QString SourceDecodeShort() { return Tr(" · не декодируется"); }

}  // namespace dlssvid
