# Hướng Dẫn: Tái Lập MSGCA_FV Best & Tìm Siêu Tham Số

## Kết quả cần tái lập

| Model | ACC | MCC | lr | dropout | Loss |
|---|---|---|---|---|---|
| **MSGCA_FV** | **0.4324 ± 0.0086** | **0.1394 ± 0.0152** | 1e-4 | 0.2 | Cross-Entropy |

> **Ghi chú quan trọng:**
> `focal_gamma=2.0` xuất hiện trong dict HP gốc nhưng **không có tác dụng**
> vì `use_focal_loss=False`. Model thực sự chỉ dùng Cross-Entropy thuần.

---

## ⚡ QUICK START — Xác nhận model chạy đúng (< 5 phút)

Chạy **trước tiên** để chắc chắn môi trường và data đúng.

```bash
# Cell 1 — Kiểm tra môi trường
python -c "
import torch, sklearn, pandas, numpy
print('torch    :', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('sklearn  :', sklearn.__version__)
print('numpy    :', numpy.__version__)
print('All OK')
"

# Cell 2 — Smoke test: 1 combo, 1 seed, 20 epochs (< 2 phút)
python run_hpsearch.py \
    --data /path/to/unified_dataset_test.pkl \
    --lr 1e-4 \
    --dropout 0.2 \
    --mode full \
    --n-seeds 1 \
    --max-epochs 20 \
    --patience 9999 \
    --warmup 5
```

**Kết quả mong đợi cell 2** (chưa converge do chỉ 20 epochs — bình thường):
```
  ACC  : ~0.38 - 0.43   (chưa ổn định, chỉ 20ep)
  MCC  : ~0.05 - 0.12
  Done. Results saved to results/
```

Nếu không crash → môi trường ổn, tiếp tục.

---

## 📋 STAGE 1 — Tái Lập Best Result (ACC=0.4324, MCC=0.1394)

**Mục tiêu:** Chạy đúng HP đã biết, đủ epochs, đủ 5 seeds.
**Ước tính:** ~90-120 phút CPU / ~20-30 phút GPU T4 (patience=9999, max_epochs=150).

### ❓ Tại sao kết quả có thể bị khác khi retrain?

**Bảng gốc** (ACC=0.4324) được tạo như sau:
```
seed=42  → LOAD từ checkpoint của main.py (train 150ep, patience=9999, best-val selection)
seed=123 → retrain từ đầu, patience=30
seed=256 → retrain từ đầu, patience=30
seed=512 → retrain từ đầu, patience=30
seed=1024→ retrain từ đầu, patience=30
avg_ep   = mean của [ep123, ep256, ep512, ep1024] → ~43 (BỎ seed=42 ra vì ep=0)
```

**Package này** (không có checkpoint seed=42):
```
seed=42  → retrain từ đầu (cần patience=9999 để match main.py)
seed=123 → retrain từ đầu
...
avg_ep   = mean của cả 5 seeds → cao hơn (80-120)
```

→ Nếu dùng `--patience 30`: seed=42 bị dừng sớm ở epoch ~50 → **MCC thấp hơn (~0.12)**
→ Nếu dùng `--patience 9999`: seed=42 train đủ 150ep → **MCC khớp (~0.14)**

---

### Cell 3 — Tạo file HP best đã biết

```bash
mkdir -p results

# Linux / macOS / Colab
echo '{"lr": 0.0001, "dropout": 0.2}' > results/msgca_best_hparams.json

# Windows PowerShell
'{"lr": 0.0001, "dropout": 0.2}' | Out-File -FilePath results\msgca_best_hparams.json -Encoding UTF8
```

### Cell 4 — Chạy final eval 5 seeds (đúng cách)

```bash
python run_hpsearch.py \
    --data /path/to/unified_dataset_test.pkl \
    --mode eval \
    --n-seeds 5 \
    --max-epochs 150 \
    --patience 9999 \
    --warmup 15 \
    --mod-dropout 0.30
```

**`--patience 9999`** (default) = không dừng sớm, chọn epoch tốt nhất trong 150 epochs → khớp với cách `main.py` chọn seed=42.

**Kết quả mong đợi:**
```
  ACC  : 0.4324 ± 0.0086     ✅ khớp với bảng gốc
  MCC  : 0.1394 ± 0.0152     ✅ khớp với bảng gốc
  avg_ep: 43
  HP    : {"lr": 0.0001, "dropout": 0.2}
```

> **Nếu kết quả lệch nhẹ:** Bình thường — variance do retrain tất cả 5 seeds,
> còn bảng gốc seed=42 được load từ checkpoint đã train sẵn.
> Quan trọng là MCC phải nằm trong khoảng `0.1394 ± 0.0152`.

---

## 🔍 STAGE 2 — Tìm HP: Search Từng Bước (Tránh Crash)

### Chiến lược chạy an toàn

Thay vì chạy tất cả 12 combos cùng lúc (dễ crash do Colab timeout),
ta **chia nhỏ từng nhóm**, mỗi cell lưu kết quả ngay lập tức.
Nếu crash → dùng `--resume` chạy lại từ điểm dừng.

---

### 🔵 Nhóm A — lr thấp (4 combos, ~25-40 phút CPU)

```bash
# Cell 5
python run_hpsearch.py \
    --data /path/to/unified_dataset_test.pkl \
    --mode search \
    --lr 5e-5 1e-4 \
    --dropout 0.1 0.2 \
    --max-epochs 100 \
    --patience 20 \
    --warmup 15 \
    --mod-dropout 0.30
```

**Kết quả lưu tại:** `results/msgca_all_results.json`
Sau đó backup ngay:
```bash
cp results/msgca_all_results.json results/group_A_results.json
```

---

### 🟡 Nhóm B — lr trung bình (thêm lr=3e-4, 2 combos mới)

```bash
# Cell 6 — Chạy TIẾP với --resume (không chạy lại nhóm A)
python run_hpsearch.py \
    --data /path/to/unified_dataset_test.pkl \
    --mode search \
    --lr 5e-5 1e-4 3e-4 \
    --dropout 0.1 0.2 \
    --resume \
    --max-epochs 100 \
    --patience 20 \
    --warmup 15
```

> `--resume` tự động bỏ qua các combo đã có trong `msgca_all_results.json`.
> Chỉ chạy **2 combo mới** (lr=3e-4, dropout=0.1 và 0.2).

---

### 🟠 Nhóm C — lr cao + dropout cao (4 combos mới)

```bash
# Cell 7
python run_hpsearch.py \
    --data /path/to/unified_dataset_test.pkl \
    --mode search \
    --lr 5e-5 1e-4 3e-4 5e-4 \
    --dropout 0.1 0.2 0.3 \
    --resume \
    --max-epochs 100 \
    --patience 20 \
    --warmup 15
```

Sau đây là tổng hợp 12 combos, `--resume` chỉ chạy các combo chưa có:
- lr=5e-4, dropout=0.1
- lr=5e-4, dropout=0.2
- lr=5e-4, dropout=0.3
- lr=3e-4, dropout=0.3
- lr=1e-4, dropout=0.3
- lr=5e-5, dropout=0.3

---

### 📊 Đọc kết quả search giữa chừng

```bash
# Cell 8 — Xem top combo sau mỗi nhóm
python -c "
import json
with open('results/msgca_all_results.json') as f:
    results = json.load(f)
results.sort(key=lambda r: r['val_mcc'], reverse=True)
print(f'Đã chạy {len(results)} combos')
print()
print(f'  {\"lr\":<10} {\"dropout\":<10} {\"val_MCC\":<12} {\"best_ep\"}')
print(f'  {\"-\"*45}')
for r in results[:5]:
    mark = \" ← BEST\" if r == results[0] else \"\"
    print(f'  {r[\"lr\"]:<10.0e} {r[\"dropout\"]:<10} {r[\"val_mcc\"]:<12.4f} ep={r[\"best_epoch\"]}{mark}')
"
```

---

## 🏁 STAGE 3 — Final Eval Với Best HP Tìm Được

Sau khi search xong, best HP được tự động lưu vào `results/msgca_best_hparams.json`.

### Cell 9 — Xem best HP

```bash
python -c "
import json
with open('results/msgca_best_hparams.json') as f:
    hp = json.load(f)
print('Best HP found:', hp)
"
```

### Cell 10 — Final eval 5 seeds với best HP

```bash
python run_hpsearch.py \
    --data /path/to/unified_dataset_test.pkl \
    --mode eval \
    --n-seeds 5 \
    --max-epochs 150 \
    --patience 30 \
    --warmup 15 \
    --mod-dropout 0.30
```

### Cell 11 — So sánh với kết quả best cũ

```bash
python -c "
import json

with open('results/msgca_final_eval.json') as f:
    r = json.load(f)

print('='*50)
print('  MSGCA_FV — Kết Quả Mới vs Bảng Gốc')
print('='*50)
print(f'  ACC: {r[\"acc_mean\"]:.4f} ± {r[\"acc_std\"]:.4f}  (gốc: 0.4324 ± 0.0086)')
print(f'  MCC: {r[\"mcc_mean\"]:.4f} ± {r[\"mcc_std\"]:.4f}  (gốc: 0.1394 ± 0.0152)')
print(f'  HP : {r[\"hparams\"]}')
print()
if r['mcc_mean'] > 0.1394:
    delta = r['mcc_mean'] - 0.1394
    print(f'  ✅ Cải thiện: +{delta:.4f} MCC so với best cũ!')
elif r['mcc_mean'] >= 0.1394 - 0.0152:
    print(f'  ✅ Trong khoảng variance — kết quả tương đương')
else:
    print(f'  ⚠️  Thấp hơn best cũ — thử search space khác')
"
```

---

## 🔄 STAGE 4 — (Tùy chọn) Thử Model Dimension Khác

Chạy riêng từng `--dim`, lưu kết quả với tên khác nhau:

### Cell 12 — Thử dim=32 (model nhỏ hơn)

```bash
python run_hpsearch.py \
    --data /path/to/unified_dataset_test.pkl \
    --mode full \
    --dim 32 \
    --lr 5e-5 1e-4 3e-4 \
    --dropout 0.1 0.2 \
    --max-epochs 100 \
    --patience 20 \
    --n-seeds 5

# Backup kết quả
cp results/msgca_final_eval.json results/final_eval_dim32.json
cp results/msgca_best_hparams.json results/best_hp_dim32.json
```

### Cell 13 — Thử dim=128 (model lớn hơn)

```bash
python run_hpsearch.py \
    --data /path/to/unified_dataset_test.pkl \
    --mode full \
    --dim 128 \
    --lr 5e-5 1e-4 \
    --dropout 0.1 0.2 \
    --max-epochs 100 \
    --patience 20 \
    --n-seeds 5

cp results/msgca_final_eval.json results/final_eval_dim128.json
cp results/msgca_best_hparams.json results/best_hp_dim128.json
```

### Cell 14 — So sánh các dim

```bash
python -c "
import json, glob, os

files = {
    'dim=32' : 'results/final_eval_dim32.json',
    'dim=64' : 'results/msgca_final_eval.json',   # default
    'dim=128': 'results/final_eval_dim128.json',
}

print(f'  {\"Config\":<10} {\"ACC\":<22} {\"MCC\":<22} {\"HP\"}')
print(f'  {\"-\"*75}')
for label, path in files.items():
    if not os.path.exists(path):
        print(f'  {label:<10} (chưa chạy)')
        continue
    with open(path) as f:
        r = json.load(f)
    hp = r.get(\"hparams\", {})
    acc = f'{r[\"acc_mean\"]:.4f}±{r[\"acc_std\"]:.4f}'
    mcc = f'{r[\"mcc_mean\"]:.4f}±{r[\"mcc_std\"]:.4f}'
    print(f'  {label:<10} {acc:<22} {mcc:<22} lr={hp.get(\"lr\",\"?\")} drop={hp.get(\"dropout\",\"?\")}')
"
```

---

## 🆘 Xử Lý Crash

### Nếu Colab disconnect giữa chừng

```bash
# Kiểm tra đã chạy được mấy combos
python -c "
import json, os
if os.path.exists('results/msgca_all_results.json'):
    with open('results/msgca_all_results.json') as f:
        r = json.load(f)
    print(f'Đã hoàn thành {len(r)} combos')
    for x in r: print(f'  lr={x[\"lr\"]:.0e} drop={x[\"dropout\"]} → MCC={x[\"val_mcc\"]:.4f}')
else:
    print('Chưa có kết quả nào — chạy lại từ đầu')
"

# Chạy lại với --resume (tự bỏ qua combo đã xong)
python run_hpsearch.py \
    --data /path/to/unified_dataset_test.pkl \
    --mode search \
    --lr 5e-5 1e-4 3e-4 5e-4 \
    --dropout 0.1 0.2 0.3 \
    --resume
```

---

## 📌 Tóm Tắt Thứ Tự Chạy

```
Cell 1  → Kiểm tra môi trường (30 giây)
Cell 2  → Smoke test 1 combo 20ep (< 2 phút)
                        ↓
Cell 3  → Tạo file HP best đã biết
Cell 4  → Tái lập ACC=0.4324 (60-90 phút CPU)
                        ↓
Cell 5  → Search nhóm A: lr=[5e-5, 1e-4] × dropout=[0.1, 0.2]
Cell 6  → Search nhóm B: thêm lr=3e-4     (--resume)
Cell 7  → Search nhóm C: thêm 5e-4+0.3   (--resume)
Cell 8  → Xem top combos
                        ↓
Cell 9  → Xem best HP
Cell 10 → Final eval 5 seeds
Cell 11 → So sánh với best cũ
                        ↓
Cell 12 → (Tùy chọn) Thử dim=32
Cell 13 → (Tùy chọn) Thử dim=128
Cell 14 → So sánh các dim
```

---

## ⏱️ Ước Tính Thời Gian

| Stage | Số combos | CPU (~) | GPU T4 (~) |
|---|---|---|---|
| Smoke test | 1 combo, 20ep | 2 phút | < 1 phút |
| Stage 1: Tái lập | 5 seeds × 43ep | 60-90 phút | 15-20 phút |
| Stage 2A: Search | 4 combos | 25-40 phút | 7-10 phút |
| Stage 2B: +2 combos | 2 combos | 12-20 phút | 3-5 phút |
| Stage 2C: +6 combos | 6 combos | 35-60 phút | 10-15 phút |
| Stage 3: Final eval | 5 seeds × 43ep | 60-90 phút | 15-20 phút |

**Tổng (full pipeline, 12 combos + final eval):** ~4-6 giờ CPU / ~50-70 phút GPU T4
