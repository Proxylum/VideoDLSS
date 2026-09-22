# Конвенции пассов (raw ↔ DLSS)

Формулы, единицы и знаки зафиксированы здесь один раз и реализованы в `core/convert/`
(`DepthConvert`, `MvConvert`, `ColorConvert`). Юнит-тесты: `tests/unit/test_depth_convert.cpp`,
`test_mv_convert.cpp`, `test_color_convert.cpp`. Форматы файлов и манифест — `core/passes/`.

## 1. Общие соглашения

- Кадры нумеруются с 0 в порядке декодирования; пасс хранит по одному файлу на кадр.
- Координаты изображения: `x` вправо, `y` вниз, начало в левом верхнем углу; центр пикселя
  `(x + 0.5, y + 0.5)`. Все векторы в пикселях того разрешения, в котором записан пасс
  (`mv.ref_width/ref_height` в манифесте, 0 = разрешение файла).
- Типы сэмплов: `u8`, `u16`, `f16` (IEEE binary16), `f32`. Целые типы хранят «сырые» значения
  (u16 цвет = round(v·65535)), без нормализации при конвертации типов.
- Внутреннее представление одного кадра пасса — `PassImage`: interleaved-каналы, построчно,
  без выравнивания.

## 2. Глубина

| Пасс | Значение | Единицы | Больше = |
|---|---|---|---|
| `depth_raw` | линейная глубина вдоль оси камеры | метры (`depth.units = meters`) или относительные (`relative = true`) | дальше |
| `depth_dlss` | reverse-Z, вход DLSS | [0, 1] | **ближе** (1 = near plane, 0 = far plane) |

Reverse-Z с конечной far-плоскостью (`near`, `far` — метры, `0 < near < far`, из манифеста):

```
d = near · (far − z) / (z · (far − near))            z ∈ [near, far] → d ∈ [1, 0]
z = near · far / (d · (far − near) + near)            обратное преобразование
```

Проверки: `z = near → d = 1`, `z = far → d = 0`, `z = 2·near·far/(near+far) → d = 0.5`;
округление float32 даёт относительную ошибку round-trip < 2·10⁻⁵ (тест).

Правила:
- `z` вне `[near, far]` клампится; `z ≤ 0`, NaN, ±inf → `d = 0` (far).
- Относительная глубина: `z_m = near + (v − min) / (max − min) · (far − near)`; `min`/`max`
  берутся из манифеста (`depth.min/max`), при нулях вычисляются по первому кадру и записываются
  в манифест, чтобы обратное преобразование было точным. `max == min` → всё на near.
- Обратное преобразование `depth_dlss → depth_raw` при `relative = true` возвращает исходные
  относительные единицы.

## 3. Motion vectors

| Пасс | Что хранит в кадре `t` | Направление | Единицы |
|---|---|---|---|
| `mv_raw` | forward optical flow `F_t` | `p(t+1) = p(t) + F_t(p(t))` (t → t+1) | пиксели исходного разрешения |
| `mv_dlss` | backward-векторы `B_t` | `p(t−1) = p(t) + B_t(p(t))` (t → t−1) | пиксели целевого (render) разрешения |

Оба — 2 канала `u` (x) и `v` (y), `y` вниз (`mv.y_up = false`). Если внешний пасс задан с `y`
вверх, установите `mv.y_up = true` — при импорте ось переворачивается (`FlipMvY`).

Конвертация `mv_raw` → `mv_dlss` (кадр `t+1` получается из flow кадра `t`):

1. **Инверсия (splat вперёд).** Для каждого пикселя `q` кадра `t`: `p = round(q + F_t(q))`;
   если `p` внутри кадра, `B_{t+1}(p) = −F_t(q)`. Коллизии (несколько `q` попали в один `p`)
   разрешаются по глубине кадра `t` — **ближайшая** поверхность (меньшее `depth_raw`) побеждает;
   без глубины побеждает последний записанный.
2. **Заполнение дыр.** Пиксели, в которые никто не попал (открывшиеся области), получают вектор
   ближайшего заполненного пикселя (BFS от заполненных).
3. **Дилатация по границам глубины** (радиус 1–2 px, `--dilate`): пиксель берёт вектор
   ближайшего по глубине соседа в окне, если тот ближе более чем на `depthEdgeThreshold`
   (2 % относительной глубины). Так DLSS получает «толстые» векторы у краёв объектов.
4. **Масштаб** под целевое разрешение `(W', H')`: nearest-ресемпл и умножение `u·W'/W`, `v·H'/H`.

Первый кадр последовательности не имеет предыдущего — его `mv_dlss` нулевой; последний кадр не имеет следующего — его `mv_raw` нулевой (стадия flow, этап 3). Обратная
конвертация `mv_dlss → mv_raw` — та же инверсия (кадр `t−1` из `B_t`), последний кадр нулевой.
Для чистого сдвига обе конвертации точны и взаимно обратны (тест).

### 3.1. Источники flow (этап 3)

| Бэкенд | Что | Единицы на выходе |
|---|---|---|
| `ofa` | NVIDIA Optical Flow Accelerator (`nvofapi64.dll`, CUDA), вход NV12 прямо из NVDEC или ABGR8 из CPU; выход `NV_OF_FLOW_VECTOR` S10.5 на сетке grid×grid (запрашивается 1) | `flow / 32` → пиксели полного разрешения; сетка > 1 апсемплируется билинейно; cost-буфер → confidence = 1 − cost/255 |
| `searaft` | SEA-RAFT (Spring-M, 4 итерации) через TensorRT из ONNX `(image1, image2: 1×3×H×W RGB 0..255) → flow 1×2×H×W`; геометрия — исходное разрешение, ограниченное `--max-res` по короткой стороне и кратное 8 | пиксели геометрии модели, масштабируются под источник (`ScaleMv`) |

Warp-тест (ТЗ §8): `WarpBackward(prev, mv_dlss_t)(p) = prev(p + mv_dlss_t(p))` должен совпадать с `cur`;
PSNR считается по валидным сэмплам (внутри кадра). TAE с компенсацией движения:
`mean |d_t(p) − d_{t−1}(p + mv_dlss_t(p))| / mean d_{t−1}`.

## 4. Цвет

`color_source` из декодированного YUV 4:2:0 8-бит: display-referred RGB (sRGB/BT.709
передаточная функция, без линеаризации), диапазон [0, 1]. Матрица и диапазон — из метаданных
потока, иначе BT.709 для ≥ 720 строк и BT.601 ниже; ограниченный диапазон:

```
Y' = (Y − 16) / 219,  Cb = (U − 128) / 224,  Cr = (V − 128) / 224      (full range: /255)
R' = Y' + 2(1−Kr)·Cr
G' = Y' − 2Kb(1−Kb)/Kg·Cb − 2Kr(1−Kr)/Kg·Cr
B' = Y' + 2(1−Kb)·Cb
BT.709: Kr = 0.2126, Kb = 0.0722;  BT.601: Kr = 0.299, Kb = 0.114;  Kg = 1 − Kr − Kb
```

Хрома реплицируется на блок 2×2 (без интерполяции) — результат детерминирован и обратим
поблочно. Значения клампятся в [0, 1]. Использованные матрица и диапазон записываются в
`stage_params.matrix` / `stage_params.range` манифеста. Линейный (scene-referred) цвет и тонмаппинг —
стадии 5–6; ключ `colorspace` в манифесте тогда будет `linear`.

## 5. Форматы файлов

| Формат | Расширение | Типы | Каналы | Примечания |
|---|---|---|---|---|
| OpenEXR | `.exr` | f16, f32 (u8→f16, u16→f32 при записи, восстанавливаются по манифесту) | по именам: `R,G,B`, `Z`, `u,v`, `A` | ZIP-сжатие, без потерь. Multi-layer: `depth.Z`, `mv.u`, `mv.v` |
| PNG | `.png` | u8, u16 | 1–4 (gray, gray+alpha, RGB, RGBA) | big-endian на диске; float не пишется — конвертировать явно |
| TIFF | `.tif` | u8, u16, f16, f32 | 1–4, contiguous | Deflate, без потерь |
| NumPy | `.npz` / `.npy` | u8, u16, f16, f32 | массив `(H, W)` или `(H, W, C)`, C-order, little-endian | ключ = имя пасса; запись stored, чтение stored + deflate (`savez_compressed`) |
| Бинарь | `.r32`, `.rg16f`, `.r16f`, `.rg32f`, `.r8` … | по расширению | по расширению | без заголовка, размер из манифеста |

## 6. Манифест (`manifest.json`, schema_version 1)

```json
{
  "schema_version": 1,
  "pass": "depth_raw",
  "width": 1920, "height": 1080,
  "fps": {"num": 24000, "den": 1001},
  "frame_count": 240, "frame_range": {"first": 0, "last": 239},
  "format": "exr", "file_pattern": "depth_raw_%06d.exr",
  "pixel_type": "f32", "channels": ["Z"],
  "colorspace": "",
  "convention": "raw",
  "model": {"name": "depth-anything-3", "version": "1.0"},
  "source": {"file": "clip.mp4", "hash": "sha256:…"},
  "stage_params": {},
  "depth": {"units": "meters", "relative": false, "near": 0.1, "far": 1000.0, "min": 0, "max": 0},
  "mv": {"direction": "forward", "y_up": false, "ref_width": 0, "ref_height": 0},
  "fingerprint": "sha256:…",
  "inputs": {"color_sr": "sha256:…", "depth_dlss": "sha256:…", "mv_dlss": "sha256:…"},
  "tool": {"app": "0.1.0", "backend": "ngx", "model": "", "model_hash": "", "dll_file": "nvngx_dlssnr.dll", "dll": "sha256:…"},
  "params_canonical": {"backend": "ngx", "intensity": 1.4},
  "created": "2026-09-22T14:09:31Z"
}
```

Последние пять полей (этап 9) пишет только `dlssvid process`: `fingerprint` — отпечаток прогона, по которому пасс
переиспользуется (`docs/architecture.md`, «Отпечатки и версии пассов»), `inputs` — отпечатки прочитанных пассов,
`tool` — версия приложения, бэкенд, модель и sha256 NVIDIA DLL, `params_canonical` — параметры стадии в
канонической записи, `created` — время завершения (UTC). У пассов, записанных `dlssvid <stage>` вручную или до
этапа 9, полей нет — такой пасс «текущий» и усыновляется первым `process`. Предыдущие версии пасса лежат рядом:
`<pass>.v/<YYYYMMDD-HHMMSS_<fp8>>/` (тот же формат папки).

Импорт без манифеста: папка сканируется по нумерованным файлам одного формата, пользователь
указывает `--pass` (и `--convention`, `--size WxH` для бинарных дампов); `--write-manifest`
сохраняет восстановленный манифест. При импорте проверяются разрешение, число кадров и
непрерывность нумерации — несовпадение перечисляется в ошибке.

## 7. CLI

```
dlssvid export  -i clip.mp4 -o out/color [--range 0-299] [--format exr|png16|tiff|npz|raw] [--preset nuke|comfyui|rawdlss]
dlssvid export  --from-dir out/depth_raw -o out/depth_npz --format npz
dlssvid export  -i clip.mp4 -o out/nuke --preset nuke --depth-dir out/depth_raw --mv-dir out/mv_raw
dlssvid import  -i out/depth_raw [--expect-size 1920x1080 --expect-frames 300] [--to-dlss out/depth_dlss --near 0.1 --far 1000]
dlssvid convert -i out/mv_raw -o out/mv_dlss --to mv_dlss --target 3840x2160 --depth-dir out/depth_raw --dilate 1
dlssvid passes  list|use <pass|stage> <version>|gc [--keep N] [--dry-run]  --passes out | --project clip.dlssvid.json   (этап 9: версии пассов)
```
