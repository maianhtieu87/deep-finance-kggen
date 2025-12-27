import torch
from torch import nn
from sklearn.metrics import accuracy_score, matthews_corrcoef

from Multimodel_Source_Encoding import MultimodalSourceEncoding
from Gated_Cross_Feature_Fusion import CrossAttentionEncoder
from Fine_grained_Movement_Prediction import FinegrainedMovementPrediction


class StockMovementModel(nn.Module):
    """
    Stock Movement Prediction Model
    Loss function: Multi-class Cross Entropy (Eq. 31)
    """

    def __init__(
        self,
        price_dim,
        macro_dim,
        dim,
        input_dim,
        hidden_dim,
        output_dim,
        num_head,
        device
    ):
        super().__init__()

        # ===== 1. Multimodal Source Encoding =====
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim,
            macro_dim=macro_dim,
            dim=dim
        )

        # ===== 2. Gated Cross-feature Fusion =====
        self.cross_attention = CrossAttentionEncoder(
            device=device,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=dim,      # ⚠ đảm bảo feature dim nhất quán
            num_head=num_head
        )

        # ===== 3. Fine-grained Movement Prediction =====
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim,
            num_classes=output_dim   # = 3
        )

        # ===== 4. Multi-class Cross Entropy Loss =====
        self.loss_fn = nn.CrossEntropyLoss()

        self.device = device

    def forward(self, s_o, s_h, s_c, s_m, label=None, mode="train"):
        """
        Inputs:
        - s_o, s_h, s_c : (B, T, 1)
        - s_m           : (B, T, num_macro)
        - label         : [[y], [y], ...] or Tensor(B,)
                          y ∈ {0: down, 1: flat, 2: up}
        """

        # ===== 1. Multimodal Encoding =====
        v_m, v_i = self.multimodal_encoder(s_o, s_h, s_c, s_m)
        # v_m, v_i : (B, T, dim)

        # ===== 2. Cross-attention Fusion =====
        fused_features, _ = self.cross_attention(v_i, v_m)
        # fused_features : (B, T, dim)

        # ===== 3. Prediction Head =====
        output_score = self.movement_predictor(
            fused_features, v_i
        )
        # output_score : (B, 3)

        # ---- numerical safety (optional but recommended) ----
        output_score = torch.clamp(output_score, -15, 15)

        # ===== TRAIN =====
        if mode == "train":
            if isinstance(label, list):
                target = torch.tensor(
                    [item[0] for item in label],
                    dtype=torch.long,
                    device=self.device
                )
            else:
                target = label.long().to(self.device)

            loss = self.loss_fn(output_score, target)
            return loss

        # ===== TEST =====
        elif mode == "test":
            if isinstance(label, list):
                target = torch.tensor(
                    [item[0] for item in label],
                    dtype=torch.long,
                    device=self.device
                )
            else:
                target = label.long().to(self.device)

            preds = torch.argmax(output_score, dim=1)

            acc = accuracy_score(
                target.cpu().numpy(),
                preds.cpu().numpy()
            )

            mcc = matthews_corrcoef(
                target.cpu().numpy(),
                preds.cpu().numpy()
            )

            return acc, mcc
