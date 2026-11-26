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
}  // namespace

QString ColourNoteText(const SourceColor& c, const ColourNote& n) {
    const RawTags& e = c.effective;
    if (n.key == "source.dv_removed") return Tr("Dolby Vision %1 снимается при любой записи: остаётся базовый слой").arg(DoviName(c));
    if (n.key == "source.dv_record_short" || n.key == "source.dv_unknown")
        return Tr("Dolby Vision %1: база не названа — цвет по тегам; Dolby Vision снимается при любой записи").arg(DoviName(c));
    if (n.key == "source.dv_ipt")
        return c.iptAssumed ? Tr("База Dolby Vision не названа, а теги цвета не указаны — принята база IPT: без RPU её цвета неверны, а RPU эта "
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

QString SourceColourText(const SourceColor& c, QString* tip) {
    bool planNote = false;
    for (const ColourNote& n : c.notes) planNote = planNote || n.plan;
    if (tip) tip->clear();
    if (c.legacyPath && !c.doviConf && !c.doviRpu && c.hdr.Empty() && !c.hdr10Plus && !planNote) return QString();
    const RawTags& e = c.effective;
    QString text;
    // what cannot be read right comes first and alone: the rest would read as fine
    if (c.refuseAll == ColourRefusal::DolbyVisionIpt) {
        text = c.iptAssumed ? Tr("Dolby Vision %1 (база не названа, принята IPT): цвета при обработке будут неверны").arg(DoviName(c))
                            : Tr("Dolby Vision %1 (база IPT): цвета при обработке будут неверны").arg(DoviName(c));
    } else if (c.refuseAll == ColourRefusal::RgbHdr) {
        text = c.rgbLayout ? Tr("RGB-поток HDR / BT.2020: читается как 8-битный SDR, цвета будут неверны")
                           : Tr("Матрица RGB при YUV-раскладке (HDR / BT.2020): цвета будут неверны");
    } else if (c.refuseProcessing == ColourRefusal::Matrix) {
        text = Tr("%1: цвета при обработке будут неверны").arg(MatrixText(e.matrix));
    } else {
        QStringList parts;
        // HDR10: PQ of BT.2020 (said or assumed) with its static metadata; a P3 or BT.709 master is «HDR (PQ)» with its gamut beside
        const bool hdr10 = c.cls == SourceClass::Pq && (e.primaries == 9 || e.primaries == 2) && (c.hdr.mdcv || c.hdr.cll);
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
        if (!c.legacyPath)
            lines << Tr("Обработка пока читает этот ролик как SDR: цвета обработанного результата будут неверны");
        for (const ColourNote& n : c.notes) lines << ColourNoteText(c, n);
        *tip = lines.join('\n');
    }
    return text;
}

}  // namespace dlssvid
