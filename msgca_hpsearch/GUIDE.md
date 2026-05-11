# Hướng Dẫn: Tái Lập MSGCA_FV Best & Tìm Siêu Tham Số

## Kết quả cần tái lập

| Model | ACC | MCC | lr | dropout | Loss |
|---|---|---|---|---|---|
| **MSGCA_FV** | **0.4324 ± 0.0086** | **0.1394 ± 0.0152** | 1e-4 | 0.2 | Cross-Entropy |

> Đây là model "fair comparison" — CE loss, không dùng focal loss, không dùng class weights.
> `focal_gamma=2.0` xuất hiện trong dict HP nhưng **không có tác dụng** vì `use_focal_loss=False`.

---

## Phần 1 — Tái Lập Kết Quả Best (ACC=0.4324)

### Bước 1: Chuẩn bị

```bash
# Cài dependencies (chỉ cần 4 packages)
pip install torch scikit-learn pandas numpy

# Đảm bảo bạn có file dataset
# Copy từ: deep-finance-kggen/data/processed/unified_dataset_test.pkl
```

### Bước 2: Tạo file HP best đã biết

Thay vì search lại, ta dùng thẳng HP đã biết:

```bash
# Tạo thư mục results nếu chưa có
mkdir -p results

# Tạo file HP (Windows PowerShell)
'{"lr": 0.0001, "dropout": 0.2}' | Out-File -FilePath results\msgca_best_hparams.json -Encoding UTF8

# Linux / Colab / macOS
echo '{"lr": 0.0001, "dropout": 0.2}' > results/msgca_best_hparams.json
```

### Bước 3: Chạy final evaluation (5 seeds)

```bash
python run_hpsearch.py \
    --data /path/to/unified_dataset_test.pkl \
    --mode eval \
    --n-seeds 5
```

**Kết quả mong đợi:**
```
  ACC  : 0.4324 ± 0.0086
  MCC  : 0.1394 ± 0.0152
  avg_ep: 43
  HP    : {"lr": 0.0001, "dropout": 0.2}
```

> **Lưu ý về variance:** Kết quả có thể lệch nhỏ do `seed=42` của MSGCA_FV best
> trong bảng gốc được load từ model đã train sẵn (không retrain).
> Chạy `--mode eval` sẽ retrain toàn bộ 5 seeds → variance ± std có thể khác nhẹ.

---

## Phần 2 — Tìm Siêu Tham Số (HP Search)

### Cách hoạt động của quá trình tìm HP

```
Grid Search (seed=42, nhanh)
│
├── Với mỗi combo (lr, dropout):
│   ├── Phase 1: Train trên train_hval → early stopping → tìm best_epoch
│   └── Evaluate trên val_hval → ghi lại val_MCC
│
└── Chọn combo có val_MCC cao nhất → best_hparams.json

Final Evaluation (5 seeds, chậm hơn)
│
└── Với best HP, chạy 5 seeds:
    ├── Phase 1: train_hval → best_epoch
    └── Phase 2: train_full → eval test → báo cáo mean±std
```

### 2.1 — Search nhanh (tái lập HP cũ)

Đây là search space **y chang** `run_experiments.py` gốc — sẽ tìm lại đúng `lr=1e-4, dropout=0.2`:

```bash
python run_hpsearch.py \
    --data /path/to/unified_dataset_test.pkl \
    --mode search \
    --lr 5e-5 1e-4 5e-4 \
    --dropout 0.1 0.2
```

**6 combos**, mỗi combo ~40-100 epochs, tổng ~20-40 phút trên CPU / ~5-10 phút trên GPU T4.

### 2.2 — Search mở rộng (tìm HP tốt hơn)

Thêm các giá trị mới vào search space:

```bash
python run_hpsearch.py \
    --data /path/to/unified_dataset_test.pkl \
    --mode full \
    --lr 5e-5 1e-4 3e-4 5e-4 \
    --dropout 0.1 0.2 0.3
```

**12 combos** — tìm kiếm thêm `lr=3e-4` và `dropout=0.3`.

### 2.3 — Search với model dimension khác nhau

Chạy **nhiều lần** với `--dim` khác nhau để tìm kích thước model tốt nhất:

```bash
# Thử dim=32 (model nhỏ hơn, ít overfit hơn)
python run_hpsearch.py \
    --data /path/to/data.pkl \
    --mode full \
    --dim 32 \
    --lr 5e-5 1e-4 3e-4 \
    --dropout 0.1 0.2

# Thử dim=128 (model lớn hơn, cần nhiều data hơn)
python run_hpsearch.py \
    --data /path/to/data.pkl \
    --mode full \
    --dim 128 \
    --lr 5e-5 1e-4 \
    --dropout 0.1 0.2
```

> **Lưu ý:** Kết quả của mỗi `--dim` sẽ ghi đè `results/msgca_best_hparams.json`.
> Đổi tên file kết quả trước khi chạy lần tiếp theo:
> ```bash
> cp results/msgca_final_eval.json results/final_eval_dim32.json
> ```

### 2.4 — Tinh chỉnh vùng HP đã tìm được

Sau khi search rộng, thu hẹp lại để tìm chính xác hơn:

```bash
# Giả sử search rộng cho thấy lr=3e-4 là tốt nhất
# Tìm tinh chỉnh quanh vùng đó:
python run_hpsearch.py \
    --data /path/to/data.pkl \
    --mode full \
    --lr 2e-4 3e-4 4e-4 \
    --dropout 0.15 0.2 0.25 \
    --n-seeds 5
```

### 2.5 — Resume nếu bị crash

```bash
# Chạy lại — tự động bỏ qua các combo đã có kết quả
python run_hpsearch.py \
    --data /path/to/data.pkl \
    --mode search \
    --lr 5e-5 1e-4 3e-4 5e-4 \
    --dropout 0.1 0.2 0.3 \
    --resume
```

---

## Phần 3 — Bảng Các HP Có Thể Thay Đổi

### HP ảnh hưởng đến model performance (nên search)

| HP | Giá trị hiện tại (best) | Range nên thử | Ảnh hưởng |
|---|---|---|---|
| `--lr` | `1e-4` | `[5e-5, 1e-4, 3e-4, 5e-4]` | Cao nhất — learning rate AdamW |
| `--dropout` | `0.2` | `[0.1, 0.2, 0.3]` | Cao — regularization |
| `--dim` | `64` | `[32, 64, 128]` | Trung bình — model capacity |
| `--num-head` | `2` | `[1, 2, 4]` | Thấp — attention heads |
| `--mod-dropout` | `0.30` | `[0.0, 0.15, 0.30]` | Trung bình — news robustness |

### HP ảnh hưởng đến training protocol (ít thay đổi)

| HP | Giá trị hiện tại | Ghi chú |
|---|---|---|
| `--max-epochs` | `150` | Tăng nếu model chưa converge |
| `--patience` | `30` | Early stopping — giảm nếu muốn search nhanh hơn |
| `--warmup` | `15` | Warmup epochs — giữ nguyên thường tốt |
| `--window-size` | `20` | Số ngày lookback — thay đổi cần rebuild data |
| `--n-seeds` | `5` | Dùng `1` cho search nhanh, `5` cho final report |

### HP cố định — **không thay đổi** để giữ fair comparison

| HP | Giá trị | Lý do |
|---|---|---|
| Loss function | Cross-Entropy | Fair comparison với baselines |
| Class weights | None | Fair comparison với baselines |
| `--train-ratio` | `0.70` | Khớp với split trong bảng gốc |
| `--valid-ratio` | `0.15` | Khớp với split trong bảng gốc |
| `--price-mode` | `vol_adjusted` | Khớp với bảng gốc |
| `--label-mode` | `rolling` | Khớp với bảng gốc |

---

## Phần 4 — Quy Trình Khuyến Nghị

```
Bước 1: Tái lập best đã biết (verify)
    → --mode eval, HP = {lr: 1e-4, dropout: 0.2}
    → Confirm ACC ≈ 0.43, MCC ≈ 0.14

Bước 2: Search rộng để tìm vùng HP tốt
    → --mode search, --lr 5e-5 1e-4 3e-4 5e-4 --dropout 0.1 0.2 0.3
    → Xem val_MCC của từng combo

Bước 3: Tinh chỉnh quanh vùng tốt nhất
    → --mode full với search space hẹp hơn
    → Final eval với 5 seeds → báo cáo mean±std

Bước 4: (Tùy chọn) Thử dim khác
    → Chạy lại với --dim 32 hoặc --dim 128
    → So sánh final MCC

Bước 5: Cập nhật bảng kết quả
    → Nếu tìm được HP cho MCC > 0.1394 → cập nhật rq1_table.txt
```

---

## Phần 5 — Output Files

```
results/
├── msgca_best_hparams.json    ← {"lr": ..., "dropout": ...}
├── msgca_all_results.json     ← Kết quả mọi combo (dùng cho --resume)
└── msgca_final_eval.json      ← {"acc_mean": ..., "mcc_mean": ..., ...}
```

**Đọc kết quả final:**
```python
import json
with open("results/msgca_final_eval.json") as f:
    r = json.load(f)
print(f"ACC: {r['acc_mean']:.4f} ± {r['acc_std']:.4f}")
print(f"MCC: {r['mcc_mean']:.4f} ± {r['mcc_std']:.4f}")
print(f"Best HP: {r['hparams']}")
```

---

## Phần 6 — Chạy Trên Google Colab

1. Zip thư mục này: `msgca_hpsearch.zip`
2. Copy file dataset: `unified_dataset_test.pkl`
3. Mở `colab_hpsearch.ipynb` → chọn **Runtime T4 GPU**
4. Chạy từng cell theo thứ tự
5. Cell cuối download `results.zip` về máy

**Hoặc chạy thủ công trên Colab:**
```python
# Cell 1: Mount và upload
from google.colab import files, drive
uploaded = files.upload()   # upload msgca_hpsearch.zip
!unzip -q msgca_hpsearch.zip
uploaded2 = files.upload()  # upload unified_dataset_test.pkl
DATA = list(uploaded2.keys())[0]

# Cell 2: Install + run
!pip install -q scikit-learn pandas numpy
!python run_hpsearch.py --data {DATA} --mode full --n-seeds 5
```
