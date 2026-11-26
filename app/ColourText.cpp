#include "ColourText.h"

#include <QCoreApplication>
#include <QStringList>

#include <cmath>

namespace dlssvid {

namespace {
QString Tr(const char* s) { return QCoreApplication::translate("ColourText", s); }
QString Q(const std::string& s) { return QString::fromStdString(s); }

QString GamutText(int primaries) {
    switch (primaries) {
        case 1: return "BT.709";
        case 9: return "BT.2020";
        case 11: return "DCI-P3";
        case 12: return "Display P3";
        case 5: case 6: return "BT.601";
        case 2: return QString();
        default: return Q(PrimariesName(primaries));
    }
}

QString MatrixText(int matrix) {
    switch (matrix) {
        case 10: return "BT.2020 CL";
        case 14: return "ICtCp (BT.2100)";
        default: return Q(MatrixName(matrix));
    }
}

QString LayersText(const SourceColor& c) {  // as LayersWords
    const std::string_view w = LayersWords(c);
    return w == "the first frame says" ? Tr("Первый кадр") : w == "the container says" ? Tr("Контейнер") : Tr("Первый кадр и контейнер");
}
QString LayersTextInside(const SourceColor& c) {  // the same within a sentence
    QString t = LayersText(c);
    if (!t.isEmpty()) t[0] = t[0].toLower();
    return t;
}

QString FieldText(std::string_view field) {
    if (field == "matrix") return Tr("матрица");
    if (field == "primaries") return Tr("охват");
    return Tr("кривая");
}

// «кривая arib-std-b67 / bt2020-10, охват ...» - FFmpeg's names of the values
QString DifferencesText(const std::vector<TagDifference>& d) {
    QStringList parts;
    for (const TagDifference& x : d)
        parts << QString("%1 %2 / %3").arg(FieldText(x.field), Q(TagValueName(x.field, x.a)), Q(TagValueName(x.field, x.b)));
    return parts.join(", ");
}

QString Nits(double v) { return QString::number(std::round(v * 10.0) / 10.0, 'g', 6); }

// «8.4»: profile and the base layer's compatibility, as Dolby Vision's names go
QString DoviName(const SourceColor& c) {
    if (c.doviConf) {  // the profiles with variants by their base (8.1, 8.4, 10.0); the others by their number (5, 7)
        const int p = c.doviConf->profile;
        return p == 8 || p == 9 || p == 10 ? QString("%1.%2").arg(p).arg(c.doviConf->blCompat) : QString::number(p);
    }
    return c.doviProfile ? QString::number(c.doviProfile) : Tr("неизвестного профиля");
}

QString FromText(TagFrom from) {
    switch (from) {
        case TagFrom::Container: return Tr("контейнер");
        case TagFrom::Stream: return Tr("поток");
        case TagFrom::Frame: return Tr("первый кадр");
        case TagFrom::Packet: return Tr("первый пакет");
        case TagFrom::None: break;
    }
    return Tr("не указано");
}

// A class as the row says it: wide gamut by the matrix alone (BT.2020's matrix under other primaries) is not «BT.2020 SDR»
QString ClassText(SourceClass c, int primaries);

QString ClassText(SourceClass c) {
    switch (c) {
        case SourceClass::Legacy: return "SDR";
        case SourceClass::Pq: return Tr("HDR (PQ)");
        case SourceClass::Hlg: return Tr("HDR HLG");
        case SourceClass::Wide: return Tr("BT.2020 SDR");
        case SourceClass::Log: return Tr("лог-кривая");
    }
    return "SDR";
}

QString ClassText(SourceClass c, int primaries) { return c == SourceClass::Wide && primaries != 9 ? Tr("SDR, матрица BT.2020") : ClassText(c); }

// TASK-0079: «Обработка ролика …» - the class as it reads after «ролика»
QString ClipClassText(const SourceColor& c) {
    if (c.cls == SourceClass::Log) return Tr("с лог-кривой");
    if (c.cls == SourceClass::Wide && c.effective.primaries != 9) return Tr("SDR с матрицей BT.2020");
    return ClassText(c.cls, c.effective.primaries);
}

// what processing waits for (ColourStop::NotYet), by the class: «обработка %1 — в следующей версии»
QString NotYetText(const SourceColor& c) {
    if (c.Note("source.layers_differ") || c.cls == SourceClass::Legacy) return Tr("такого ролика");  // layers that disagree
    switch (c.cls) {
        case SourceClass::Pq:
        case SourceClass::Hlg: return Tr("HDR");
        case SourceClass::Wide: return Tr("широкого охвата");
        case SourceClass::Log: return Tr("лог-кривой");
        case SourceClass::Legacy: break;
    }
    return Tr("такого ролика");
}

// passthrough writes the effective tags - and the HDR records of PQ / HLG only
QString AsItIsText(const SourceColor& c) { return c.Hdr() ? Tr("как есть, с HDR") : Tr("как есть, с его тегами цвета"); }
}  // namespace

QString ColourNoteText(const SourceColor& c, const ColourNote& n) {
    const RawTags& e = c.effective;
    if (n.key == "source.dv_removed") return Tr("Dolby Vision %1 снимается при любой записи: остаётся базовый слой").arg(DoviName(c));
    if (n.key == "source.dv_record_short")
        return Tr("Dolby Vision %1: запись не называет базу (старая или короче 5 байт) — цвет по тегам; Dolby Vision снимается при любой записи")
            .arg(DoviName(c));
    if (n.key == "source.dv_unknown")
        return Tr("Dolby Vision %1: база не распознана — цвет по тегам; Dolby Vision снимается при любой записи").arg(DoviName(c));
    if (n.key == "source.dv_ipt")
        return c.iptAssumed ? Tr("База Dolby Vision не распознана, а матрица и кривая не указаны — принята база IPT: без RPU её цвета неверны, а RPU эта "
                                 "программа не применяет")
                            : Tr("База Dolby Vision в IPT: без RPU цвета неверны, а RPU эта программа не применяет — цвета при обработке будут неверны");
    if (n.key == "source.matrix_unsupported")
        return Tr("Матрица %1 читается как BT.709 / BT.601 — цвета при обработке будут неверны").arg(MatrixText(e.matrix));
    if (n.key == "source.rgb_hdr")
        return c.rgbLayout ? Tr("RGB-поток HDR / BT.2020 читается как 8-битный SDR — цвета будут неверны")
                           : Tr("Поток YUV помечен матрицей RGB (0) — ни одна матрица YUV к нему не подходит, цвета будут неверны");
    if (n.key == "source.bt2020_assumed") return Tr("Матрица или охват не указаны — принят BT.2020");
    if (n.key == "source.trc_unspecified") return Tr("Охват BT.2020 без кривой — принята кривая SDR");
    if (n.key == "source.trc_not_converted") return Tr("Кривая %1 при охвате BT.2020 (лог камеры?) не переводится").arg(Q(TrcName(e.trc)));
    if (n.key == "source.matrix_2020")
        return e.primaries == 2 ? Tr("Матрица BT.2020, охват не указан — принят широкий охват")
                                : Tr("Матрица BT.2020 при охвате %1 — принят широкий охват").arg(GamutText(e.primaries));
    if (n.key == "source.av1_stream_metadata")
        return Tr("Метаданные HDR внутри потока AV1 здесь не читаются и в результат не переносятся") +
               (c.hdr.Empty() ? Tr("; в контейнере их нет") : Tr("; в контейнере есть"));
    if (n.key == "source.no_hdr_metadata") return Tr("PQ без метаданных мастеринга и уровня света");
    if (n.key == "source.hdr10plus") return Tr("Динамические метаданные HDR10+ в результат не переносятся");
    if (n.key == "source.layers_differ")
        return Tr("%1 (%2) и параметры кодека (%3) размечены по-разному")
            .arg(LayersText(c), ClassText(c.cls, c.effective.primaries), ClassText(c.rawCls, c.stream.primaries));
    if (n.key == "source.frame_differs") return Tr("Первый кадр и поток размечены по-разному (кадр / поток): %1").arg(DifferencesText(FrameDifferences(c)));
    if (n.key == "source.container_differs")
        return Tr("Контейнер размечен иначе, взяты поток или кадр (контейнер / взято): %1").arg(DifferencesText(ContainerDifferences(c)));
    if (n.key == "source.probe_failed")
        return c.lookFailed == FirstLook::Packet ? Tr("Первый видеопакет не прочитан — HDR10+ не проверен")
                                                  : Tr("Первый кадр не прочитан — цвет только по тегам потока и контейнера, записи HDR кадра не прочитаны");
    return Q(n.detail);
}

QString ColourStopShortText(ColourStop stop, const SourceColor& c) {
    switch (stop) {
        case ColourStop::None: return QString();
        case ColourStop::NotYet: return Tr("обработка %1 — в следующей версии").arg(NotYetText(c));
        case ColourStop::Matrix: return Tr("обработка этой матрицы не поддерживается");
        case ColourStop::DolbyVisionIpt:
        case ColourStop::RgbHdr: return Tr("цвет источника не поддерживается");
        case ColourStop::NoContainerTags: return Tr("контейнер не хранит теги цвета");
    }
    return QString();
}

QString ColourStopText(ColourStop stop, const SourceColor& c) {
    const QString passthrough = Tr("«Обработать» без стадий («только кодирование») запишет ролик %1").arg(AsItIsText(c));
    switch (stop) {
        case ColourStop::None: return QString();
        case ColourStop::NotYet:
            if (c.Note("source.layers_differ"))
                return Tr("Слои ролика размечены по-разному (%1: %2; параметры кодека: %3): обработка такого ролика появится в следующей версии. %4")
                    .arg(LayersTextInside(c), ClassText(c.cls, c.effective.primaries), ClassText(c.rawCls, c.stream.primaries), passthrough);
            return Tr("Обработка ролика %1 появится в следующей версии: сейчас его цвета вышли бы неверными. %2").arg(ClipClassText(c), passthrough);
        case ColourStop::Matrix:
            return Tr("Матрица %1: обработка не поддерживается. %2").arg(MatrixText(c.effective.matrix), passthrough);
        case ColourStop::DolbyVisionIpt:
            return c.iptAssumed ? Tr("Dolby Vision %1: база не распознана, а матрица и кривая не указаны — принята база IPT; без RPU её цвета неверны — "
                                     "ни обработка, ни запись как есть")
                                      .arg(DoviName(c))
                                : Tr("Dolby Vision %1 с базой IPT не поддерживается: без RPU её цвета неверны — ни обработка, ни запись как есть").arg(DoviName(c));
        case ColourStop::RgbHdr:
            return c.rgbLayout ? Tr("RGB-поток HDR / BT.2020 не поддерживается: он читается как 8-битный SDR — ни обработка, ни запись как есть")
                               : Tr("Поток YUV помечен матрицей RGB: ни одна матрица YUV к нему не подходит — ни обработка, ни запись как есть");
        case ColourStop::NoContainerTags:  // the GUI keeps the project's result extension: it is set with the project (project --result)
            return Tr("Этот контейнер не хранит теги цвета: ролик %1 пишется только в .mp4, .mov или .mkv — укажите такой результат проекта")
                .arg(ClipClassText(c));
    }
    return QString();
}

QString PlanNoteText(const SourceColor& c, const PlanNote& n) {
    if (n.scope == "source") return ColourNoteText(c, ColourNote{n.key, n.text, true});
    if (n.key == "encode.tags_filled") {
        const ColorTags t = WrittenTags(c);
        return c.Hdr() ? Tr("Неуказанные теги цвета записываются как BT.2020, чтобы плееры узнали HDR")
                       : Tr("Неуказанные теги цвета записываются по классу источника: матрица %1, охват %2, кривая %3")
                             .arg(Q(MatrixName(t.matrix)), Q(PrimariesName(t.primaries)), Q(TrcName(t.transfer)));
    }
    if (n.key == "encode.hdr_8bit") return Tr("HDR в 8 битах: плавные переходы станут ступеньками — лучше HEVC или AV1 10 бит, или ProRes");
    if (n.key == "encode.range_lost") return Tr("Полный диапазон: результат его не хранит — плееры и декодеры прочтут ограниченный");
    if (n.key == "encode.mdcv_lost") return Tr("Данные мастеринга неполные, а этот контейнер пишет их только целиком — они не перенесутся");
    if (n.key == "encode.mdcv_unfit") return Tr("Данные мастеринга не помещаются в поля этого контейнера — они не перенесутся");
    if (n.key == "encode.mdcv_invalid")
        return Tr("В данных мастеринга нулевая координата или яркость max не выше min — читатели этого контейнера отбросят эту часть (или их целиком)");
    if (n.key == "encode.cll_invalid")
        return Tr("Уровень света (MaxCLL / MaxFALL) с одним нулём или вне полей этого контейнера — он не перенесётся");
    if (n.key == "encode.no_amve") return Tr("Условия просмотра (ambient viewing environment) в этот контейнер не переносятся");
    return Q(n.text);
}

QString SourceColourText(const SourceColor& c, QString* tip) {
    bool planNote = false;
    for (const ColourNote& n : c.notes) planNote = planNote || n.plan;
    if (tip) tip->clear();
    if (c.legacyPath && !c.doviConf && !c.doviRpu && c.hdr.Empty() && !c.hdr10Plus && !planNote) return QString();
    const RawTags& e = c.effective;
    QString text;
    // what cannot be read right comes first and alone: the rest would read as fine
    if (c.refuseAll == ColourRefusal::DolbyVisionIpt) {
        text = c.iptAssumed ? Tr("Dolby Vision %1 (база не распознана, принята IPT) — не поддерживается").arg(DoviName(c))
                            : Tr("Dolby Vision %1 (база IPT) — не поддерживается").arg(DoviName(c));
    } else if (c.refuseAll == ColourRefusal::RgbHdr) {
        text = c.rgbLayout ? Tr("RGB-поток HDR / BT.2020 — не поддерживается")
                           : Tr("Матрица RGB при YUV-раскладке (HDR / BT.2020) — не поддерживается");
    } else if (c.refuseProcessing == ColourRefusal::Matrix) {
        text = Tr("%1 — обработка не поддерживается").arg(MatrixText(e.matrix));
    } else {
        QStringList parts;
        // HDR10: PQ of BT.2020 (said or assumed) with its static metadata; a P3 or BT.709 master is «HDR (PQ)» with its gamut beside
        const bool hdr10 = c.cls == SourceClass::Pq && (e.primaries == 9 || e.primaries == 2) && (c.hdr.mdcv || c.hdr.cll) &&
                           (c.bitDepth == 0 || c.bitDepth >= 10);  // 8-bit PQ is no HDR10
        // wide gamut by the matrix alone (BT.2020's matrix under other primaries): said as such, the primaries beside it
        parts << (hdr10 ? Tr("HDR10 (PQ)") : ClassText(c.cls, e.primaries));
        const QString gamut = GamutText(e.primaries);
        if (!gamut.isEmpty() && !(c.cls == SourceClass::Wide && e.primaries == 9)) parts << gamut;  // «BT.2020 SDR» says it already
        else if (c.Hdr()) parts << Tr("BT.2020 (не указан)");
        if (c.hdr.mdcv && c.hdr.mdcv->hasLuminance) parts << Tr("мастеринг %1 нит").arg(Nits(c.hdr.mdcv->maxLum.Value()));
        if (c.hdr.cll) parts << QString("MaxCLL %1/%2").arg(c.hdr.cll->maxCll).arg(c.hdr.cll->maxFall);
        if (c.doviConf || c.doviRpu) parts << Tr("Dolby Vision %1 (будет снят)").arg(DoviName(c));
        if (c.hdr10Plus) parts << Tr("HDR10+ (не переносится)");
        text = parts.join(" · ");
    }
    if (tip) {
        QStringList lines;
        lines << Tr("Матрица %1 (%2), охват %3 (%4), кривая %5 (%6), диапазон %7 (%8)")
                     .arg(Q(MatrixName(e.matrix)), FromText(c.matrixFrom), Q(PrimariesName(e.primaries)), FromText(c.primariesFrom), Q(TrcName(e.trc)),
                          FromText(c.trcFrom), Q(RangeName(e.range)), FromText(c.rangeFrom));
        if (c.refuseAll != ColourRefusal::None)
            lines << Tr("Ни обработка, ни запись как есть: такой ролик эта версия не пишет");
        else if (c.refuseProcessing == ColourRefusal::Matrix)  // no later version lifts it
            lines << Tr("Обработка этой матрицы не поддерживается; без стадий («только кодирование») ролик пишется %1").arg(AsItIsText(c));
        else if (!c.legacyPath)
            lines << Tr("Обработка %1 — в следующей версии; без стадий («только кодирование») ролик пишется %2").arg(NotYetText(c), AsItIsText(c));
        for (const ColourNote& n : c.notes) lines << ColourNoteText(c, n);
        *tip = lines.join('\n');
    }
    return text;
}

}  // namespace dlssvid
