# imgdiff

工业大图（BMP，8-bit/16-bit，灰度/彩色）前后差异检测工具：先做高精度对齐（支持轻微旋转/平移），再输出差异mask与热力图叠加图，便于复判。

## 构建

```bash
cmake -S . -B build
cmake --build build -j
```

生成可执行文件：`build/imgdiff`

## 用法

单对图片：

```bash
./build/imgdiff --ref A.bmp --tgt B.bmp --out out --prefix sample1
```

文件夹批处理（按同名bmp配对）：

```bash
./build/imgdiff --ref_dir before --tgt_dir after --out out
```

常用参数：

```bash
./build/imgdiff \
  --ref A.bmp --tgt B.bmp --out out \
  --motion euclidean \
  --max_dim 2000 \
  --k_mad 6 \
  --min_area 30
```

输出（示例，以 `sample1` 为前缀）：
- `sample1_overlay.png`：差异热力图叠加
- `sample1_mask.png`：差异二值mask
- `sample1_regions.csv`：差异区域列表（外接矩形+面积）
- `sample1_tgt_aligned.png`：对齐后的目标图（用于快速检查对齐质量）
- `sample1_report.json`：warp矩阵、阈值、差异像素数等
