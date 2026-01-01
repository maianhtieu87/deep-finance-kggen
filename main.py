import torch
import random
import numpy as np
from src.model import StockMovementModel
from src.data_loader import data_prepare
from configs.config import TrainConfig

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

device = torch.device(
    "cuda" if TrainConfig.use_cuda and torch.cuda.is_available() else "cpu"
)
set_seed(TrainConfig.seed)

def train_model(data):
    s_o = data["s_o"].to(device)
    s_h = data["s_h"].to(device)
    s_c = data["s_c"].to(device)
    s_m = data["s_m"].to(device)
    label = data["label"]

    model = StockMovementModel(
        price_dim=1,
        macro_dim=s_m.shape[-1],
        dim=TrainConfig.dim,
        input_dim=TrainConfig.input_dim,
        hidden_dim=TrainConfig.hidden_dim,
        output_dim=TrainConfig.output_dim,
        num_head=TrainConfig.num_head,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TrainConfig.learning_rate,
        weight_decay=TrainConfig.weight_decay
    )

    best_acc, best_mcc = 0, 0

    for epoch in range(TrainConfig.epoch_num):
        model.train()
        optimizer.zero_grad()

        loss = model(s_o, s_h, s_c, s_m, label, mode="train")
        loss.backward()
        assert torch.isfinite(loss), "❌ LOSS IS NaN"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            acc, mcc = model(s_o, s_h, s_c, s_m, label, mode="test")

        print(f"Epoch {epoch} | Loss {loss.item():.4f} | ACC {acc:.4f} | MCC {mcc:.4f}")

        best_acc = max(best_acc, acc)
        best_mcc = max(best_mcc, mcc)

    print(f"\nBEST → ACC: {best_acc:.4f}, MCC: {best_mcc:.4f}")

if __name__ == "__main__":
    dp = data_prepare(r"D:\Project\NCKH\data\env_data\synchronized_data.pkl")

    train_data, _ = dp.prepare_data(
        stock_name="TSLA",
        window_size=20,
        future_days=1,
        train_ratio=0.8
    )

    train_model(train_data)
