import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

class AsoftMax(nn.Module):
    """
    P2SGrad Loss для anti-spoofing.
    Реализация из статьи ASVspoof2019.
    """

    def __init__(self, margin=0.2, scale=15):
        super().__init__()
        # Параметры будут установлены в forward
        self.smooth = 0.1
        self.m_loss = nn.MSELoss()

    def smooth_labels(self, labels):
        factor = self.smooth
        # smooth the labels
        labels *= (1 - factor)
        labels += (factor / labels.shape[1])
        return labels

    def forward(self, batch, **kwargs):
        """
        P2SGrad loss compute
        
        Args:
            batch (dict): batch containing 'logits' and 'labels'
            **kwargs: дополнительные аргументы (игнорируются)
        Returns:
            losses (dict): dictionary loss
        """
        input_feat = batch['logits']
        target = batch['labels']
        
        # Получаем размеры
        batchsize, in_dim = input_feat.shape
        out_dim = 2  # 2 класса для anti-spoofing
        
        # Создаем learnable weights если их нет
        if not hasattr(self, 'weight'):
            self.weight = Parameter(torch.Tensor(in_dim, out_dim))
            self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        
        # normalize the weight (again)
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        
        # normalize the input feature vector
        x_modulus = input_feat.pow(2).sum(1).pow(0.5)
        
        # W * x = ||W|| * ||x|| * cos()
        inner_wx = input_feat.mm(w)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        # P2Sgrad MSE
        target = target.long()
        
        # filling in the target
        with torch.no_grad():
            index = torch.zeros_like(cos_theta)
            index.scatter_(1, target.data.view(-1, 1), 1)
            index = self.smooth_labels(index)
    
        # MSE between \cos\theta and one-hot vectors
        loss = self.m_loss(cos_theta, index)
        
        return {"loss": loss}