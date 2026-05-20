import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import math

# ============================================================================
# 方案1: 改进的直接回归方法（修复原始问题）
# ============================================================================

class ImprovedDirectRegression(nn.Module):
    """
    相比简单linear的改进：
    1. 多层MLP with residual connections
    2. LayerNorm稳定训练
    3. Dropout防止过拟合
    4. 坐标归一化到[-1,1]而不是原始像素值
    """
    def __init__(self, feature_dim: int, hidden_dims=[512, 256, 128], dropout=0.1):
        super().__init__()
        
        layers = []
        in_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 最后输出层，分别预测x和y
        self.coord_head = nn.Linear(hidden_dims[-1], 2)
        
        # 可选：预测置信度（帮助过滤低质量预测）
        self.confidence_head = nn.Linear(hidden_dims[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """关键：好的初始化避免梯度消失"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, Fea] 扁平化特征
        Returns:
            coords: [B, 2] 归一化坐标，范围[-1, 1]
            confidence: [B, 1] 预测置信度
        """
        feat = self.feature_extractor(x)
        coords = self.coord_head(feat)  # [B, 2]
        coords = torch.tanh(coords)  # 限制到[-1, 1]
        confidence = torch.sigmoid(self.confidence_head(feat))  # [B, 1]
        
        return coords, confidence


class SmoothL1LossWithConfidence(nn.Module):
    """
    Smooth L1 Loss（对离群点更鲁棒）+ 置信度加权
    """
    def __init__(self, beta=1.0, conf_weight=0.1):
        super().__init__()
        self.beta = beta
        self.conf_weight = conf_weight
    
    def forward(self, pred_coords, pred_conf, target_coords, valid_mask=None):
        """
        Args:
            pred_coords: [B, 2] 预测坐标（归一化）
            pred_conf: [B, 1] 预测置信度
            target_coords: [B, 2] 真实坐标（归一化）
            valid_mask: [B] 标记哪些样本有效
        """
        # Smooth L1 for coordinates
        diff = torch.abs(pred_coords - target_coords)
        loss_coord = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        if valid_mask is not None:
            loss_coord = loss_coord * valid_mask.unsqueeze(1)
        
        loss_coord = loss_coord.mean()
        
        # Binary cross-entropy for confidence
        # 如果有效样本，目标置信度为1；否则为0
        target_conf = valid_mask.float().unsqueeze(1) if valid_mask is not None else torch.ones_like(pred_conf)
        loss_conf = F.binary_cross_entropy(pred_conf, target_conf)
        
        return loss_coord + self.conf_weight * loss_conf


# ============================================================================
# 方案2: 热力图方法（主流，类似CenterNet）
# ============================================================================

class HeatmapDecoder(nn.Module):
    """
    将flatten特征重建为空间热力图
    """
    def __init__(self, 
                 feature_dim: int,
                 spatial_size: Tuple[int, int] = (8, 8),  # encoder输出的空间大小
                 output_size: Tuple[int, int] = (64, 64),  # 热力图大小
                 num_channels: int = 512):
        super().__init__()
        
        self.spatial_size = spatial_size
        self.output_size = output_size
        self.feature_channels = num_channels
        
        # 1. 将flatten特征reshape回空间维度
        expected_dim = num_channels * spatial_size[0] * spatial_size[1]
        if feature_dim != expected_dim:
            # 如果维度不匹配，先做一次映射
            self.dim_adapter = nn.Linear(feature_dim, expected_dim)
        else:
            self.dim_adapter = nn.Identity()
        
        # 2. 上采样网络（转置卷积）
        self.upsample_layers = nn.ModuleList()
        in_channels = num_channels
        
        # 计算需要多少次上采样：8x8 -> 64x64 需要3次x2上采样
        num_upsample = int(math.log2(output_size[0] // spatial_size[0]))
        
        for i in range(num_upsample):
            out_channels = in_channels // 2 if i < num_upsample - 1 else 64
            self.upsample_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 
                                      kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels
        
        # 3. 最终输出层
        self.heatmap_head = nn.Conv2d(64, 1, kernel_size=1)  # 热力图
        self.offset_head = nn.Conv2d(64, 2, kernel_size=1)   # 子像素偏移
        
    def forward(self, x):
        """
        Args:
            x: [B, Fea] 扁平化特征
        Returns:
            heatmap: [B, 1, H, W] 热力图
            offset: [B, 2, H, W] 每个位置的偏移量
        """
        B = x.size(0)
        
        # Reshape to spatial
        x = self.dim_adapter(x)
        x = x.view(B, self.feature_channels, self.spatial_size[0], self.spatial_size[1])
        
        # Upsample
        for layer in self.upsample_layers:
            x = layer(x)
        
        # Output heads
        heatmap = torch.sigmoid(self.heatmap_head(x))  # [B, 1, H, W]
        offset = self.offset_head(x)  # [B, 2, H, W]
        
        return heatmap, offset


class FocalLossWithOffset(nn.Module):
    """
    Focal Loss for heatmap + L1 Loss for offset
    """
    def __init__(self, alpha=2, beta=4, offset_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.offset_weight = offset_weight
    
    def forward(self, pred_heatmap, pred_offset, target_heatmap, target_offset, offset_mask):
        """
        Args:
            pred_heatmap: [B, 1, H, W]
            pred_offset: [B, 2, H, W]
            target_heatmap: [B, 1, H, W] 高斯热力图
            target_offset: [B, 2, H, W] 真实偏移
            offset_mask: [B, 1, H, W] 标记哪些位置需要计算offset loss
        """
        # Focal loss for heatmap
        pred_heatmap = torch.clamp(pred_heatmap, min=1e-4, max=1-1e-4)
        
        pos_mask = target_heatmap.eq(1).float()  # 正样本位置
        neg_mask = target_heatmap.lt(1).float()  # 负样本位置
        
        # 正样本loss
        pos_loss = -((1 - pred_heatmap) ** self.alpha) * torch.log(pred_heatmap) * pos_mask
        
        # 负样本loss（带权重衰减）
        neg_weight = (1 - target_heatmap) ** self.beta
        neg_loss = -(pred_heatmap ** self.alpha) * torch.log(1 - pred_heatmap) * neg_weight * neg_mask
        
        heatmap_loss = (pos_loss + neg_loss).sum() / (pos_mask.sum() + 1e-4)
        
        # L1 loss for offset（只在有目标的位置计算）
        offset_loss = F.l1_loss(pred_offset * offset_mask, 
                                target_offset * offset_mask, 
                                reduction='sum')
        offset_loss = offset_loss / (offset_mask.sum() + 1e-4)
        
        total_loss = heatmap_loss + self.offset_weight * offset_loss
        
        return total_loss, heatmap_loss, offset_loss


def generate_gaussian_heatmap(coords, heatmap_size, sigma=2):
    """
    生成高斯热力图
    WWWW
    Args:
        coords: [B, 2] 归一化坐标 [-1, 1]W
        heatmap_size: (H, W)
        sigma: 高斯核标准差（像素单位）
    Returns:
        heatmap: [B, 1, H, W]
        offset: [B, 2, H, W]
        offset_mask: [B, 1, H, W]
    """
    B = coords.size(0)
    H, W = heatmap_size
    
    # 将归一化坐标转换为像素坐标
    coords_pixel = (coords + 1) / 2 * torch.tensor([W-1, H-1], device=coords.device)
    
    heatmap = torch.zeros(B, 1, H, W, device=coords.device)
    offset = torch.zeros(B, 2, H, W, device=coords.device)
    offset_mask = torch.zeros(B, 1, H, W, device=coords.device)
    
    for i in range(B):
        x, y = coords_pixel[i]
        x_int, y_int = int(x), int(y)
        
        # 生成高斯分布
        for dy in range(-3*sigma, 3*sigma+1):
            for dx in range(-3*sigma, 3*sigma+1):
                xx, yy = x_int + dx, y_int + dy
                if 0 <= xx < W and 0 <= yy < H:
                    g = math.exp(-((dx**2 + dy**2) / (2 * sigma**2)))
                    heatmap[i, 0, yy, xx] = max(heatmap[i, 0, yy, xx], g)
        
        # 在中心点位置计算offset
        if 0 <= x_int < W and 0 <= y_int < H:
            offset[i, 0, y_int, x_int] = x - x_int  # x方向偏移
            offset[i, 1, y_int, x_int] = y - y_int  # y方向偏移
            offset_mask[i, 0, y_int, x_int] = 1
    
    return heatmap, offset, offset_mask


def extract_peak_from_heatmap(heatmap, offset, k=1, threshold=0.3):
    """
    从热力图中提取峰值位置
    
    Args:
        heatmap: [B, 1, H, W]
        offset: [B, 2, H, W]
        k: 提取top-k个峰值
        threshold: 置信度阈值
    Returns:
        coords: [B, k, 2] 提取的坐标（归一化）
        scores: [B, k] 置信度
    """
    B, _, H, W = heatmap.shape
    
    # Max pooling找局部最大值
    max_pooled = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    keep = (heatmap == max_pooled).float()
    heatmap = heatmap * keep
    
    # Flatten并找top-k
    heatmap_flat = heatmap.view(B, -1)
    scores, indices = torch.topk(heatmap_flat, k, dim=1)
    
    # 转换为坐标
    y_coords = (indices // W).float()
    x_coords = (indices % W).float()
    
    # 加上offset进行亚像素精度修正
    for i in range(B):
        for j in range(k):
            if scores[i, j] > threshold:
                y_int, x_int = int(y_coords[i, j]), int(x_coords[i, j])
                x_coords[i, j] += offset[i, 0, y_int, x_int]
                y_coords[i, j] += offset[i, 1, y_int, x_int]
    
    # 归一化到[-1, 1]
    coords = torch.stack([
        x_coords / (W - 1) * 2 - 1,
        y_coords / (H - 1) * 2 - 1
    ], dim=-1)
    
    return coords, scores


# ============================================================================
# 评估指标
# ============================================================================

class LocalizationMetrics:
    """
    定位任务的评估指标集合
    """
    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size
        self.reset()
    
    def reset(self):
        self.errors = []  # 欧式距离误差（像素）
        self.correct_at_thresholds = {1: 0, 2: 0, 5: 0, 10: 0}  # 不同阈值下的正确数
        self.total = 0
    
    def update(self, pred_coords, target_coords):
        """
        Args:
            pred_coords: [B, 2] 归一化坐标 [-1, 1]
            target_coords: [B, 2] 归一化坐标 [-1, 1]
        """
        B = pred_coords.size(0)
        
        # 转换为像素坐标
        pred_pixel = (pred_coords + 1) / 2 * torch.tensor(
            [self.image_size[1]-1, self.image_size[0]-1], 
            device=pred_coords.device
        )
        target_pixel = (target_coords + 1) / 2 * torch.tensor(
            [self.image_size[1]-1, self.image_size[0]-1],
            device=target_coords.device
        )
        
        # 计算欧式距离
        distances = torch.sqrt(((pred_pixel - target_pixel) ** 2).sum(dim=1))
        
        self.errors.extend(distances.cpu().numpy().tolist())
        
        for threshold in self.correct_at_thresholds.keys():
            self.correct_at_thresholds[threshold] += (distances < threshold).sum().item()
        
        self.total += B
    
    def compute(self) -> Dict[str, float]:
        """计算所有指标"""
        errors = np.array(self.errors)
        
        metrics = {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(errors)),
            'min_error': float(np.min(errors)),
        }
        
        # PCK (Percentage of Correct Keypoints) at different thresholds
        for threshold, count in self.correct_at_thresholds.items():
            metrics[f'PCK@{threshold}px'] = count / self.total if self.total > 0 else 0
        
        return metrics
    
    def __str__(self):
        metrics = self.compute()
        s = "Localization Metrics:\n"
        s += f"  Mean Error: {metrics['mean_error']:.2f} px\n"
        s += f"  Median Error: {metrics['median_error']:.2f} px\n"
        s += f"  Std Error: {metrics['std_error']:.2f} px\n"
        for threshold in [1, 2, 5, 10]:
            s += f"  PCK@{threshold}px: {metrics[f'PCK@{threshold}px']*100:.2f}%\n"
        return s


# ============================================================================
# 训练示例
# ============================================================================

def train_direct_regression_example():
    """方案1：改进的直接回归训练示例"""
    
    # 模型
    model = ImprovedDirectRegression(feature_dim=2048, hidden_dims=[512, 256, 128])
    criterion = SmoothL1LossWithConfidence()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # 假设数据
    for epoch in range(100):
        # 假设从dataloader获取
        features = torch.randn(32, 2048)  # [B, Fea]
        target_coords = torch.rand(32, 2) * 2 - 1  # [B, 2] 范围[-1, 1]
        valid_mask = torch.ones(32)  # [B] 所有样本都有效
        
        # 前向
        pred_coords, pred_conf = model(features)
        loss = criterion(pred_coords, pred_conf, target_coords, valid_mask)
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


def train_heatmap_example():
    """方案2：热力图方法训练示例"""
    
    # 模型
    model = HeatmapDecoder(
        feature_dim=2048,
        spatial_size=(8, 8),
        output_size=(64, 64)
    )
    criterion = FocalLossWithOffset(alpha=2, beta=4, offset_weight=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    metrics = LocalizationMetrics(image_size=(256, 256))
    
    for epoch in range(100):
        # 假设数据
        features = torch.randn(32, 2048)
        target_coords = torch.rand(32, 2) * 2 - 1  # 归一化坐标
        
        # 生成目标热力图
        target_heatmap, target_offset, offset_mask = generate_gaussian_heatmap(
            target_coords, 
            heatmap_size=(64, 64),
            sigma=2
        )
        
        # 前向
        pred_heatmap, pred_offset = model(features)
        loss, heatmap_loss, offset_loss = criterion(
            pred_heatmap, pred_offset,
            target_heatmap, target_offset, offset_mask
        )
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 评估
        if epoch % 10 == 0:
            with torch.no_grad():
                pred_coords, scores = extract_peak_from_heatmap(
                    pred_heatmap, pred_offset, k=1, threshold=0.3
                )
                metrics.update(pred_coords.squeeze(1), target_coords)
                
                print(f"Epoch {epoch}")
                print(f"  Total Loss: {loss.item():.4f}")
                print(f"  Heatmap Loss: {heatmap_loss.item():.4f}")
                print(f"  Offset Loss: {offset_loss.item():.4f}")
                print(metrics)
                metrics.reset()


class ComprehensiveLocalizationMetrics:
    """综合评估指标"""
    
    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size
        self.reset()
    
    def reset(self):
        self.errors = []
        self.oks_scores = []
        self.pck_counts = {1: 0, 2: 0, 5: 0, 10: 0}
        self.total = 0
    
    def update(self, pred_coords, target_coords, scales=None):
        """
        Args:
            pred_coords: [B, 2] 归一化坐标 [-1, 1]
            target_coords: [B, 2] 归一化坐标 [-1, 1]
            scales: [B] 可选，目标尺度
        """
        B = pred_coords.size(0)
        
        # 转换为像素坐标
        pred_pixel = (pred_coords + 1) / 2 * torch.tensor(
            [self.image_size[1]-1, self.image_size[0]-1], 
            device=pred_coords.device
        )
        target_pixel = (target_coords + 1) / 2 * torch.tensor(
            [self.image_size[1]-1, self.image_size[0]-1],
            device=target_coords.device
        )
        
        # 1. 欧式距离误差
        distances = torch.sqrt(((pred_pixel - target_pixel) ** 2).sum(dim=1))
        self.errors.extend(distances.cpu().numpy().tolist())
        
        # 2. PCK at different thresholds
        for threshold in self.pck_counts.keys():
            self.pck_counts[threshold] += (distances < threshold).sum().item()
        
        # 3. OKS (如果提供了尺度信息)
        if scales is not None:
            oks = torch.exp(-(distances ** 2) / (2 * scales ** 2 * 4.0))
            self.oks_scores.extend(oks.cpu().numpy().tolist())
        
        self.total += B
    
    def compute(self):
        errors = np.array(self.errors)
        
        metrics = {
            # 距离指标
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            
            # PCK指标
            'PCK@1px': self.pck_counts[1] / self.total,
            'PCK@2px': self.pck_counts[2] / self.total,
            'PCK@5px': self.pck_counts[5] / self.total,
            'PCK@10px': self.pck_counts[10] / self.total,
        }
        
        # OKS指标
        if self.oks_scores:
            metrics['mean_OKS'] = float(np.mean(self.oks_scores))
            metrics['AP@OKS0.5'] = float(np.mean([s > 0.5 for s in self.oks_scores]))
            metrics['AP@OKS0.75'] = float(np.mean([s > 0.75 for s in self.oks_scores]))
        
        return metrics

# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("方案1: 改进的直接回归")
    print("=" * 60)
    train_direct_regression_example()
    
    print("\n" + "=" * 60)
    print("方案2: 热力图方法")
    print("=" * 60)
    train_heatmap_example()