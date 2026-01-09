```
[RENDER_START]──────────► Rendering ──────────►[RENDER_END]
                                                     │
                                                     ├──► Color Conversion
                                                     │
                                                     └──► H.264 Encoding ──►[ENCODE_END]
                                                     │                           │
                                                     └────── Render-to-Encode ───┘
└───────────────────── Total Pipeline ───────────────────────────────────────────┘
```

headless
```
[INFO]  ========== Latency Statistics (100 frames) ==========
[INFO]  Render Time:        avg=0.02ms, min=0.02ms, max=0.02ms
[INFO]  Color Conversion:   avg=0.04ms, min=0.04ms, max=0.04ms
[INFO]  Encode Time:        avg=0.46ms, min=0.45ms, max=0.51ms
[INFO]  Render-to-Encode:   avg=0.64ms, min=0.63ms, max=0.71ms  <-- KEY
[INFO]  Total Pipeline:     avg=0.66ms, min=0.64ms, max=0.73ms
[INFO]  ====================================================
```

headed
```
[INFO]  ========== Latency Statistics (100 frames) ==========
[INFO]  Render Time:        avg=0.01ms, min=0.01ms, max=0.02ms
[INFO]  Color Conversion:   avg=0.07ms, min=0.04ms, max=0.09ms
[INFO]  Encode Time:        avg=0.44ms, min=0.44ms, max=0.46ms
[INFO]  Render-to-Encode:   avg=0.67ms, min=0.64ms, max=0.75ms  <-- KEY
[INFO]  Total Pipeline:     avg=0.68ms, min=0.65ms, max=0.76ms
[INFO]  ====================================================
```


after optimizations:
0A: Removed Render Wait from Headless Loop
0B: Pre-Record Command Buffer
`[INFO]    Average FPS: 1461.99`

```
[INFO]  ========== Latency Statistics (100 frames) ==========
[INFO]  Render Time:        avg=0.02ms, min=0.01ms, max=0.02ms
[INFO]  Color Conversion:   avg=0.04ms, min=0.04ms, max=0.04ms
[INFO]  Encode Time:        avg=0.44ms, min=0.44ms, max=0.49ms
[INFO]  Render-to-Encode:   avg=0.59ms, min=0.56ms, max=0.66ms  <-- KEY
[INFO]  Total Pipeline:     avg=0.60ms, min=0.58ms, max=0.68ms
[INFO]  ====================================================
```