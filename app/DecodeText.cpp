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
            if (s.part)  // TASK-0076: a part of the video in another format (a new sequence header)
                return Tr("Видео AV1 не декодируется на этом компьютере: в нём есть часть, формат которой (%1) эта версия программы не "
                          "декодирует — AV1 декодирует только видеокарта, и только 8 и 10 бит 4:2:0. Что сделать: %2.")
                    .arg(LayoutText(s), transcode);
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
                      "версии программы нет. Что сделать: переустановите или обновите драйвер NVIDIA и перезапустите программу или %1.")
                .arg(transcode);
        case DecodeBlock::NoAv1:
            if (s.part)
                return Tr("Видео AV1 не декодируется на этом компьютере: в нём есть часть AV1 %2 бит, а такой AV1 видеокарта %1 не декодирует, и "
                          "другого декодера AV1 в этой версии программы нет. Что сделать: %3 или откройте его на компьютере с видеокартой, которая "
                          "декодирует AV1 %2 бит (например, RTX 30 и новее).")
                    .arg(gpu)
                    .arg(s.bitDepth)
                    .arg(transcode);
            return Tr("Видео AV1 не декодируется на этом компьютере: видеокарта %1 не декодирует AV1 %2 бит, а другого декодера AV1 в этой версии "
                      "программы нет. Что сделать: %3 или откройте его на компьютере с видеокартой, которая декодирует AV1 (например, RTX 30 и новее).")
                .arg(gpu)
                .arg(s.bitDepth)
                .arg(transcode);
        case DecodeBlock::Size: {  // «кадр в файле»: the stored frame - a turned video is shown the other way round
            const bool small = s.width < s.caps.minWidth || s.height < s.caps.minHeight;
            const bool large = s.width > s.caps.maxWidth || s.height > s.caps.maxHeight;
            // TASK-0076: a part of the video with another frame size (a new sequence header)
            const QString frame = (s.part ? Tr("в нём есть часть, где кадр в файле %1×%2") : Tr("кадр в файле %1×%2")).arg(s.width).arg(s.height);
            if (!small && !large)  // the macroblock count alone (FFmpeg's test: w * h / 256)
                return Tr("Видео AV1 не декодируется на этом компьютере: %1 слишком велик для видеокарты %2 — она берёт не больше %3 "
                          "блоков 16×16, а другого декодера AV1 в этой версии программы нет. Что сделать: перекодируйте ролик другой программой в "
                          "H.264 или HEVC либо в кадр поменьше.")
                    .arg(frame, gpu)
                    .arg(s.caps.maxMacroblocks);
            if (small && large)  // one side below the minimum, the other above the maximum
                return Tr("Видео AV1 не декодируется на этом компьютере: %1 вне пределов, которые декодирует видеокарта %2 (от %3×%4 до "
                          "%5×%6), а другого декодера AV1 в этой версии программы нет. Что сделать: перекодируйте ролик в H.264 или HEVC другой программой.")
                    .arg(frame, gpu)
                    .arg(s.caps.minWidth)
                    .arg(s.caps.minHeight)
                    .arg(s.caps.maxWidth)
                    .arg(s.caps.maxHeight);
            return Tr("Видео AV1 не декодируется на этом компьютере: %1 %2, чем декодирует видеокарта %3 (от %4×%5 до %6×%7), а другого "
                      "декодера AV1 в этой версии программы нет. Что сделать: %8.")
                .arg(frame, small ? Tr("меньше") : Tr("больше"), gpu)
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
