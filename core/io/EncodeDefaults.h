#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace dlssvid {

// Bitrate defaults for the lossy encoders (h264_nvenc | hevc_nvenc | av1_nvenc). Without an explicit `b` FFmpeg's NVENC
// wrapper encodes at its own low default — a 3840×1600 result came out at ~1.4 Mbit/s — so a run picks the bitrate from
// the output size, rate and codec instead: H.264 at the usual upload recommendations (720p30 ≈ 10, 1080p30 ≈ 16,
// 1440p30 ≈ 24, 4K30 ≈ 45, 4K60 ≈ 70 Mbit/s; linear between, +50 % from 30 to 60 fps), HEVC ≈ 0.65× and AV1 ≈ 0.55×
// of that. The GUI shows the same number as «Авто» and offers presets in Mbit/s; the CLI takes `--codec-opt b=50M`.
bool LossyCodec(const std::string& codec);  // false for ffv1 (lossless): no bitrate at all
double RecommendedBitrateMbps(const std::string& codec, uint32_t width, uint32_t height, double fps);
// "50M" | "50000k" | "50000000" | "1.5M" -> Mbit/s; nullopt when it is not a bitrate.
std::optional<double> ParseBitrateMbps(const std::string& ffmpegValue);
// 50 -> "50M", 1.5 -> "1500k": the value of the encoder option `b`.
std::string BitrateOption(double mbps);
// The encoder options a run uses: `b` is added from RecommendedBitrateMbps when the codec is lossy and none is given.
std::map<std::string, std::string> EffectiveCodecOptions(const std::string& codec, const std::map<std::string, std::string>& options, uint32_t width,
                                                         uint32_t height, double fps);

}  // namespace dlssvid
