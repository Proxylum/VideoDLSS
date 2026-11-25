#include "DecodeText.h"

#include <QCoreApplication>

#include "io/VideoDecoder.h"

namespace dlssvid {

namespace {
QString Tr(const char* s) { return QCoreApplication::translate("DecodeText", s); }

QString LayoutText(const DecodeSupport& s) { return DecodeLayoutText(s.bitDepth, s.chroma, s.pixelFormat); }
}  // namespace

QString DecodeLayoutText(int bitDepth, const std::string& chroma, const std::string& pixelFormat) {  // «10 бит 4:4:4», «8 бит RGB»
    if (chroma == "420" || chroma == "422" || chroma == "444")
        return Tr("%1 бит %2:%3:%4").arg(bitDepth).arg(chroma[0]).arg(chroma[1]).arg(chroma[2]);
    const std::string kind = LayoutKindOf(pixelFormat);  // TASK-0076: «other» told apart where the format says it
    if (kind == "rgb") return Tr("%1 бит RGB").arg(bitDepth);
    if (kind == "gray") return Tr("%1 бит, монохромный").arg(bitDepth);
    if (kind == "palette") return Tr("%1 бит, палитра").arg(bitDepth);
    if (kind.size() == 3) return Tr("%1 бит %2:%3:%4").arg(bitDepth).arg(kind[0]).arg(kind[1]).arg(kind[2]);  // 4:1:1, 4:4:0 ...
    return Tr("%1 бит, монохромный или особый формат").arg(bitDepth);
}

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
            if (s.pixelFormat == "unknown")  // the depth asked about is a default, not the video's (a GPU without 8-bit AV1 has no AV1)
                return Tr("Видео AV1 не декодируется на этом компьютере: видеокарта %1 не декодирует AV1, а другого декодера AV1 в этой версии "
                          "программы нет. Что сделать: %2 или откройте его на компьютере с видеокартой, которая декодирует AV1 (например, RTX 30 и "
                          "новее).")
                    .arg(gpu, transcode);
            if (s.part)
                return Tr("Видео AV1 не декодируется на этом компьютере: в нём есть часть AV1 %2 бит, а такой AV1 видеокарта %1 не декодирует, и "
                          "другого декодера AV1 в этой версии программы нет. Что сделать: %3 или откройте его на компьютере с видеокартой, которая "
                          "декодирует AV1 %2 бит (например, RTX 30 и новее).")
                    .arg(gpu)
                    .arg(s.bitDepth)
                    .arg(transcode);
            if (s.height % 2 == 1)  // the odd height, known: no GPU takes it in this build - not «open it on another computer» alone
                return Tr("Видео AV1 не декодируется на этом компьютере: видеокарта %1 не декодирует AV1 %2 бит, а другого декодера AV1 в этой "
                          "версии программы нет; к тому же высота его кадра в файле нечётная (%3×%4), а такой AV1 эта версия программы не "
                          "декодирует ни на какой видеокарте. Что сделать: %5 или в кадр с чётными сторонами — такой откроет компьютер с "
                          "видеокартой, которая декодирует AV1 (например, RTX 30 и новее).")
                    .arg(gpu)
                    .arg(s.bitDepth)
                    .arg(s.width)
                    .arg(s.height)
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
        case DecodeBlock::OddHeight:  // «кадр в файле»: the stored frame - a turned video is shown the other way round (its odd side a width)
            if (s.part)  // a part of the video with another frame size (a new sequence header)
                return Tr("Видео AV1 не декодируется на этом компьютере: в нём есть часть, где высота кадра в файле нечётная (%1×%2), а такой AV1 "
                          "видеокарта в этой версии программы выдаёт со сдвинутым цветом, другого же декодера AV1 нет. Что сделать: перекодируйте "
                          "ролик в H.264 или HEVC другой программой или в кадр с чётными сторонами.")
                    .arg(s.width)
                    .arg(s.height);
            return Tr("Видео AV1 не декодируется на этом компьютере: высота его кадра в файле нечётная (%1×%2), а такой AV1 видеокарта в этой "
                      "версии программы выдаёт со сдвинутым цветом, другого же декодера AV1 нет. Что сделать: перекодируйте ролик в H.264 или HEVC "
                      "другой программой или в кадр с чётными сторонами.")
                .arg(s.width)
                .arg(s.height);
        case DecodeBlock::Resized:  // the sequence's declared maximum, as stored (a turned video is shown the other way round)
            return Tr("Видео AV1 не декодируется на этом компьютере: в нём есть кадры меньше размера, заявленного в файле (%1×%2), а такие AV1 "
                      "эта версия программы не декодирует. Что сделать: %3.")
                .arg(s.width)
                .arg(s.height)
                .arg(transcode);
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
