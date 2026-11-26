#include "convert/ColorPlan.h"

namespace dlssvid {

ColorPlan ResolveColor(const VideoStreamInfo& info) {
    ColorPlan p;
    const ColorInfo sdr = ColorInfoFromStream(info);
    p.colourDecode = sdr;
    p.guideDecode = sdr;
    p.viewDecode = sdr;
    p.resultEncode = sdr;
    p.resultTags = ResultColorTags(p.resultEncode, info);
    return p;
}

ColorTags ResultColorTags(const ColorInfo& encode, const VideoStreamInfo& info) {
    ColorTags t;
    t.range = encode.range == ColorRange::Full ? 2 : 1;  // AVCOL_RANGE_JPEG / MPEG
    switch (encode.matrix) {
        case ColorMatrix::Bt709: t.matrix = 1; break;                                // AVCOL_SPC_BT709
        case ColorMatrix::Bt601: t.matrix = info.colorSpace == 5 ? 5 : 6; break;     // BT470BG as the source said, else SMPTE170M
        case ColorMatrix::Bt2020: t.matrix = 9; break;                               // AVCOL_SPC_BT2020_NCL
    }
    t.primaries = info.colorPrimaries;
    t.transfer = info.colorTrc;
    return t;
}

}  // namespace dlssvid
