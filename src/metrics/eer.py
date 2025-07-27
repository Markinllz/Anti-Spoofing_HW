import torch
import numpy as np
from typing import Dict, Any

from src.metrics.base_metric import BaseMetric


class EERMetric(BaseMetric):
    """
    Equal Error Rate (EER) metric for audio anti-spoofing.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: additional arguments
        """
        super(EERMetric, self).__init__()
        
        print("📈 Инициализация EERMetric...")
        
        # Логируем параметры
        for key, value in kwargs.items():
            print(f"   📊 {key}: {value}")
        
        self.name = "eer"
        print("✅ EERMetric инициализирован")

    def forward(self, **batch) -> float:
        """
        Compute EER metric.
        
        Args:
            **batch: input batch containing scores and labels
            
        Returns:
            float: EER value
        """
        # Логируем входные данные (только для отладки)
        if hasattr(self, '_debug_forward') and self._debug_forward:
            print(f"   📈 EERMetric forward: входные ключи {list(batch.keys())}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"      {key}: shape={value.shape}, dtype={value.dtype}")
        
        # Получаем scores и labels
        if 'scores' in batch:
            scores = batch['scores']
        elif 'logits' in batch:
            # Если у нас есть logits, берем вероятность второго класса (spoof)
            logits = batch['logits']
            scores = torch.softmax(logits, dim=1)[:, 1]
        else:
            print("❌ Ошибка: не найдены scores или logits в батче")
            return 0.0
        
        labels = batch['labels']
        
        # Логируем размеры
        if hasattr(self, '_debug_forward') and self._debug_forward:
            print(f"   📊 Scores: shape={scores.shape}, range=[{scores.min().item():.4f}, {scores.max().item():.4f}]")
            print(f"   📊 Labels: shape={labels.shape}, unique={torch.unique(labels).tolist()}")
        
        # Вычисляем EER
        eer = self._compute_eer(scores, labels)
        
        # Логируем результат
        if hasattr(self, '_debug_forward') and self._debug_forward:
            print(f"   📈 EER: {eer:.4f}")
        
        return eer

    def _compute_eer(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute Equal Error Rate.
        
        Args:
            scores (torch.Tensor): prediction scores
            labels (torch.Tensor): ground truth labels
            
        Returns:
            float: EER value
        """
        # Конвертируем в numpy
        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Получаем уникальные пороги
        thresholds = np.unique(scores_np)
        
        # Вычисляем FAR и FRR для каждого порога
        far_values = []
        frr_values = []
        
        for threshold in thresholds:
            # FAR = FP / (FP + TN) = FP / (FP + TN)
            # FRR = FN / (FN + TP) = FN / (FN + TP)
            
            # Предсказания: 1 если score >= threshold, иначе 0
            predictions = (scores_np >= threshold).astype(int)
            
            # Вычисляем confusion matrix
            tp = np.sum((predictions == 1) & (labels_np == 1))
            tn = np.sum((predictions == 0) & (labels_np == 0))
            fp = np.sum((predictions == 1) & (labels_np == 0))
            fn = np.sum((predictions == 0) & (labels_np == 1))
            
            # Вычисляем FAR и FRR
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            far_values.append(far)
            frr_values.append(frr)
        
        # Находим точку, где FAR ≈ FRR
        far_values = np.array(far_values)
        frr_values = np.array(frr_values)
        
        # Находим индекс, где разность минимальна
        diff = np.abs(far_values - frr_values)
        min_idx = np.argmin(diff)
        
        # EER - это среднее FAR и FRR в этой точке
        eer = (far_values[min_idx] + frr_values[min_idx]) / 2
        
        return float(eer)

    def set_debug_mode(self, debug_forward=False):
        """
        Включает режим отладки для логирования forward pass.
        
        Args:
            debug_forward (bool): логировать forward pass
        """
        self._debug_forward = debug_forward
        if debug_forward:
            print(f"🐛 Режим отладки включен для {self.__class__.__name__}")
            print(f"   📈 Debug forward: {debug_forward}")