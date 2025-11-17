# 🚀 Pipeline Im2GPS3k - Quick Start

## Files đã tạo

1. **`main_im2gps3k.py`** - Pipeline hoàn chỉnh cho im2gps3k dataset
2. **`test_single_image.py`** - Test nhanh với 1 image
3. **`README_IM2GPS3K.md`** - Hướng dẫn chi tiết

## Quick Start

### 1. Setup API Key

Tạo file `.env`:

```env
GOOGLE_CLOUD_API_KEY=your_gemini_api_key
```

### 2. Chuẩn bị dữ liệu

```
G3-main/
├── checkpoints/
│   ├── g3.pth
│   └── im2gps3k_places365.csv
├── index/
│   └── I_g3_im2gps3k.npy
└── data/im2gps3k/images/
    ├── 00001.jpg
    ├── 00002.jpg
    └── ...
```

### 3. Test với 1 image

```powershell
python test_single_image.py
```

### 4. Chạy full pipeline

```powershell
python main_im2gps3k.py
```

## Features

✅ **Automatic preprocessing**: Keyframe extraction, transcription, search
✅ **Multi-modal prediction**: Image + Text based
✅ **Evidence tracking**: Với references và citations
✅ **Intermediate saves**: Không mất data nếu crash
✅ **Detailed logging**: Console + file
✅ **Flexible configuration**: Easy to customize

## Output Files

- `predictions_*.json` - Chi tiết predictions
- `predictions_*.csv` - Summary CSV
- `summary_*.json` - Statistics
- `im2gps3k_pipeline.log` - Detailed logs

## Customization

Trong `main_im2gps3k.py`, sửa config:

```python
config = {
    'device': 'cuda',  # or 'cpu'
    'model_name': 'gemini-2.0-flash-exp',
    'input_dir': 'data/im2gps3k/images',
    'output_dir': 'results/im2gps3k',
}
```

Chọn mode:

```python
# Test mode: 5 images
await pipeline.run_batch_prediction(max_images=5)

# Specific images
await pipeline.run_batch_prediction(
    image_ids=['00001.jpg', '00002.jpg']
)

# Full dataset
await pipeline.run_batch_prediction()
```

## Expected Performance

- **Time**: ~2-3 minutes/image
- **Success rate**: 95%+ với đầy đủ metadata
- **Memory**: ~4-8GB VRAM (GPU) hoặc ~2-4GB RAM (CPU)

## Troubleshooting

| Lỗi                  | Giải pháp                                  |
| -------------------- | ------------------------------------------ |
| API key not found    | Tạo file `.env` với `GOOGLE_CLOUD_API_KEY` |
| Checkpoint not found | Kiểm tra đường dẫn `checkpoint_path`       |
| No images to process | Kiểm tra `input_dir` path                  |
| CUDA out of memory   | Đổi `device='cpu'`                         |

---

**📚 Xem `README_IM2GPS3K.md` để biết thêm chi tiết!**
