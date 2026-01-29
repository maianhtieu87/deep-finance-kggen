import torch
from torch import nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
import torch.nn.functional as F
from encoders.mutil_encoder import MultimodalSourceEncoding
from .fusion import StableGatedCrossAttention
from .predictor import FinegrainedMovementPrediction    

class FocalLoss(nn.Module):
    """
    Focal Loss: Giải quyết mất cân bằng bằng cách tập trung vào mẫu khó (Hard Examples).
    Công thức: FL(pt) = - alpha * (1 - pt)^gamma * log(pt)
    
    Args:
        alpha: Tensor trọng số cho từng class [w_0, w_1, w_2]. 
               Ví dụ: [1.5, 0.5, 1.5] nghĩa là class 0 và 2 được chú ý hơn class 1
        gamma: Độ tập trung vào mẫu khó. 
               gamma=0: giống CE loss thông thường
               gamma=2: mặc định trong paper
               gamma=1: nhẹ hơn, khuyến nghị cho imbalanced data
        reduction: 'mean', 'sum', hoặc 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (B, num_classes)
            targets: Ground truth labels (B,)
        """
        # 1. Tính Cross Entropy Loss với Weight (Alpha)
        # weight=self.alpha sẽ tự động nhân trọng số vào loss của từng lớp
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # 2. Tính pt (xác suất model dự đoán đúng class)
        pt = torch.exp(-ce_loss)
        
        # 3. Áp dụng Focal modulation: (1 - pt)^gamma
        # Khi pt cao (model tự tin đúng) → (1-pt) nhỏ → giảm loss (easy sample)
        # Khi pt thấp (model không chắc) → (1-pt) lớn → tăng loss (hard sample)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 4. Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ================================================================
# [NEW] LABEL SMOOTHING CROSS ENTROPY
# ================================================================
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy với Label Smoothing để chống overconfident.
    
    Thay vì target = [0, 1, 0] (one-hot), dùng [ε/K, 1-ε+ε/K, ε/K]
    
    Args:
        smoothing: Hệ số làm mượt (0-1). Khuyến nghị 0.1-0.2
        weight: Class weights (tương tự alpha trong Focal)
    """
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Tạo smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Áp dụng class weights nếu có
        if self.weight is not None:
            weight_expanded = self.weight.unsqueeze(0).expand_as(log_probs)
            loss = -(true_dist * log_probs * weight_expanded).sum(dim=-1)
        else:
            loss = -(true_dist * log_probs).sum(dim=-1)
            
        return loss.mean()

        
class StockMovementModel(nn.Module):
    """
    MSGCA Framework Implementation (Refactored).
    Integrates Stable Gated Fusion and MLP-based Aggregation.
    """

    def __init__(
        self,
        price_dim,
        macro_dim,
        news_dim,
        dim,
        input_dim,
        output_dim,
        num_head,
        device, 
        dropout=0.1, 
        class_weights=None,
        use_focal_loss=True,
        focal_gamma=2.0,           # [NEW] Cho phép điều chỉnh gamma
        use_label_smoothing=False, # [NEW] Option để dùng label smoothing
        smoothing=0.1,             # [NEW] Smoothing factor
    ):
        super().__init__()
        self.device = device
        self.output_dim = output_dim

        # ===== 1. Multimodal Source Encoding =====
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim,
            macro_dim=macro_dim,
            news_dim=news_dim,
            dim=dim
        )

        # ===== 2. Stable Gated Cross-Feature Fusion =====
        self.fusion_news = StableGatedCrossAttention(dim=dim, num_head=num_head)
        self.fusion_macro = StableGatedCrossAttention(dim=dim, num_head=num_head)

        # ===== 3. Fine-grained Movement Prediction (Decoder) =====
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim,
            window_size=input_dim,
            num_classes=output_dim,
            dropout=dropout
        )

        # ===== 4. Loss Function Strategy =====
        self.use_focal_loss = use_focal_loss
        self.use_label_smoothing = use_label_smoothing
        
        if use_label_smoothing:
            # Label Smoothing (good alternative to Focal)
            self.loss_fn = LabelSmoothingCrossEntropy(smoothing=smoothing, weight=class_weights)
            print(f"🔧 Loss Strategy: LABEL SMOOTHING (ε={smoothing}) + Weights={class_weights is not None} ✅")
            
        elif use_focal_loss:
            # Focal Loss với gamma điều chỉnh được
            self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)
            
            # Log chi tiết
            if class_weights is not None:
                print(f"🔧 Loss Strategy: FOCAL LOSS (γ={focal_gamma}) + ALPHA BALANCING ✅")
                print(f"   ► Alpha weights: {class_weights.cpu().numpy() if torch.is_tensor(class_weights) else class_weights}")
            else:
                print(f"🔧 Loss Strategy: FOCAL LOSS (γ={focal_gamma}) - No Alpha ⚠️")
        else:
            # Vanilla Cross Entropy
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            print(f"🔧 Loss Strategy: STANDARD CE + Weights={class_weights is not None}")

    def forward(self, s_o, s_h, s_c, s_m, s_n, label=None, mode="train", return_preds=False, return_logits=False):
        """
        Forward pass with multiple output options for debugging.
        
        Args:
            s_o, s_h, s_c: Price features (B, T, 1)
            s_m: Macro features (B, T, macro_dim)
            s_n: News features (B, T, news_dim)
            label: Ground truth labels (B,)
            mode: "train" or "test"
            return_preds: If True, return predictions in test mode
            return_logits: If True, return raw logits in test mode
            
        Returns:
            - mode="train": loss (scalar)
            - mode="test": 
                - Default: (acc, mcc)
                - With return_preds: (acc, mcc, predictions)
                - With return_logits: (acc, mcc, predictions, logits)
        """
        # 1. Encode Features
        v_m, v_i, v_n = self.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)

        # 2. Stable Fusion (Guided by Indicator v_i)
        fused_news = self.fusion_news(primary=v_i, aux=v_n)
        fused_macro = self.fusion_macro(primary=v_i, aux=v_m)
        
        # 3. Combine Fused Features
        v_fused_total = (fused_news + fused_macro) / 2.0
        
        # 4. Prediction
        logits = self.movement_predictor(fused_seq=v_fused_total, orig_seq=v_i)
        
        # Numerical stability: Clamp logits to prevent overflow
        logits = torch.clamp(logits, -15, 15)

        # ===== TRAIN MODE =====
        if mode == "train":
            if isinstance(label, list):
                target = torch.tensor([item[0] for item in label], dtype=torch.long, device=self.device)
            else:
                target = label.long().to(self.device)
            
            loss = self.loss_fn(logits, target)
            return loss

        # ===== TEST MODE =====
        elif mode == "test":
            if isinstance(label, list):
                target = torch.tensor([item[0] for item in label], dtype=torch.long, device=self.device)
            else:
                target = label.long().to(self.device)

            # Get predictions
            preds = torch.argmax(logits, dim=1)

            # Compute metrics
            acc = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
            mcc = matthews_corrcoef(target.cpu().numpy(), preds.cpu().numpy())
            
            # Return based on flags
            if return_logits:
                return acc, mcc, preds, logits
            elif return_preds:
                return acc, mcc, preds
            else:
                return acc, mcc
        
        # ===== LOGITS MODE (For Temperature Scaling Experiments) =====
        elif mode == "logits":
            return logits
    
    def get_prediction_confidence(self, s_o, s_h, s_c, s_m, s_n):
        """
        Utility method to analyze prediction confidence.
        
        Returns:
            probs: Softmax probabilities (B, num_classes)
            preds: Predicted classes (B,)
            confidence: Max probability for each sample (B,)
        """
        with torch.no_grad():
            v_m, v_i, v_n = self.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)
            fused_news = self.fusion_news(primary=v_i, aux=v_n)
            fused_macro = self.fusion_macro(primary=v_i, aux=v_m)
            v_fused_total = (fused_news + fused_macro) / 2.0
            logits = self.movement_predictor(fused_seq=v_fused_total, orig_seq=v_i)
            
            probs = F.softmax(logits, dim=-1)
            confidence, preds = torch.max(probs, dim=-1)
            
        return probs, preds, confidence