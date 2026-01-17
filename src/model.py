import torch
from torch import nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
import torch.nn.functional as F  # [NEW] Import thêm functional
from encoders.mutil_encoder import MultimodalSourceEncoding
from .fusion import StableGatedCrossAttention
from .predictor import FinegrainedMovementPrediction    

class FocalLoss(nn.Module):
    """
    Focal Loss: Giải quyết mất cân bằng bằng cách tập trung vào mẫu khó (Hard Examples).
    Công thức: FL(pt) = - (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Alpha = None nghĩa là không dùng class weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Tính CE Loss (không reduce để nhân với thừa số Focal)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Tính pt (xác suất dự báo đúng)
        pt = torch.exp(-ce_loss)
        
        # Công thức Focal: giảm trọng số mẫu dễ (pt cao)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class StockMovementModel(nn.Module):
    """
    MSGCA Framework Implementation (Refactored).
    Integrates Stable Gated Fusion and MLP-based Aggregation.
    """

    def __init__(
        self,
        price_dim,
        macro_dim,
        news_dim,           # [NEW] Input dimension của News (1024)
        dim,                # Hidden Dimension (D)
        input_dim,          # Window Size (T) - Đổi tên biến này thành window_size ở Config
        output_dim,         # Num classes (3)
        num_head,
        device, 
        dropout=0.1, 
        class_weights=None,
        use_focal_loss=True,    
    ):
        super().__init__()
        self.device = device

        # ===== 1. Multimodal Source Encoding =====
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim,
            macro_dim=macro_dim,
            news_dim=news_dim, # [NEW] Truyền news_dim vào encoder
            dim=dim
        )

        # ===== 2. Stable Gated Cross-Feature Fusion =====
        # We use Price (Indicator) as the Primary modality for both fusions.
        
        # Fusion 1: Price + News
        self.fusion_news = StableGatedCrossAttention(dim=dim, num_head=num_head)
        
        # Fusion 2: Price + Macro
        self.fusion_macro = StableGatedCrossAttention(dim=dim, num_head=num_head)

        # ===== 3. Fine-grained Movement Prediction (Decoder) =====
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim,
            window_size=input_dim, # Strictly passed as Window Size (T)
            num_classes=output_dim,
            dropout=dropout
        )

        # ===== 4. Loss Function =====
        if use_focal_loss:
            # Chiến lược: Chỉ dùng Gamma=2.0 để tự cân bằng, bỏ qua Alpha (class_weights)
            self.loss_fn = FocalLoss(alpha=None, gamma=2.0)
            print("🔧 Using Loss Strategy: FOCAL LOSS (Gamma=2.0, No Class Weights)")
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    def forward(self, s_o, s_h, s_c, s_m, s_n, label=None, mode="train"):
        """
        Forward pass with correct fusion flow.
        Inputs:
            s_n: News tensor (B, T, news_dim)
        """
        # 1. Encode Features
        # v_m, v_i, v_n: (B, T, D)
        v_m, v_i, v_n = self.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)

        # 2. Stable Fusion (Guided by Indicator v_i)
        # Price + News
        fused_news = self.fusion_news(primary=v_i, aux=v_n)
        
        # Price + Macro
        fused_macro = self.fusion_macro(primary=v_i, aux=v_m)
        
        # 3. Combine Fused Features
        # Simple averaging to get single fused representation
        v_fused_total = (fused_news + fused_macro) / 2.0
        
        # 4. Prediction
        # Pass both Fused features and Original Indicator features to Decoder
        logits = self.movement_predictor(fused_seq=v_fused_total, orig_seq=v_i)
        
        # Optional: Numerical clamp to prevent extreme logits
        logits = torch.clamp(logits, -15, 15)

        # ===== TRAIN/TEST LOGIC =====
        if mode == "train":
            if isinstance(label, list):
                target = torch.tensor([item[0] for item in label], dtype=torch.long, device=self.device)
            else:
                target = label.long().to(self.device)
            
            # CrossEntropyLoss expects Raw Logits
            loss = self.loss_fn(logits, target)
            return loss

        elif mode == "test":
            if isinstance(label, list):
                target = torch.tensor([item[0] for item in label], dtype=torch.long, device=self.device)
            else:
                target = label.long().to(self.device)

            # Convert Logits to Class Indices
            preds = torch.argmax(logits, dim=1)

            acc = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
            mcc = matthews_corrcoef(target.cpu().numpy(), preds.cpu().numpy())
            return acc, mcc