from __future__ import print_function

import numpy as np
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_f
from torch.nn import Parameter
from typing import Dict, Any


class P2SActivationLayer(torch_nn.Module):
    """ Output layer that produces cos\theta between activation vector x
    and class vector w_j

    in_dim:     dimension of input feature vectors
    output_dim: dimension of output feature vectors 
                (i.e., number of classes)
    """
    def __init__(self, in_dim, out_dim):
        super(P2SActivationLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        return

    def forward(self, input_feat):
        """
        Compute P2sgrad activation
        
        input:
        ------
          input_feat: tensor (batchsize, input_dim)

        output:
        -------
          tensor (batchsize, output_dim)
          
        """
        # normalize the weight (again)
        # w (feature_dim, output_dim)
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        
        # normalize the input feature vector
        # x_modulus (batchsize)
        # sum input -> x_modules in shape (batchsize)
        x_modulus = input_feat.pow(2).sum(1).pow(0.5)
        # w_modules (output_dim)
        # w_moduls should be 1, since w has been normalized
        w_modulus = w.pow(2).sum(0).pow(0.5)

        # W * x = ||W|| * ||x|| * cos())))))))
        # inner_wx (batchsize, output_dim)
        inner_wx = input_feat.mm(w)
        # cos_theta (batchsize, output_dim)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        # done
        return cos_theta


class P2SGradLoss(torch_nn.Module):
    """P2SGradLoss() MSE loss between output and target one-hot vectors
    
    See usage in __doc__ of P2SActivationLayer
    """
    def __init__(self):
        super(P2SGradLoss, self).__init__()
        self.m_loss = torch_nn.MSELoss()

    def forward(self, input_score, target):
        """ 
        input
        -----
          input_score: tensor (batchsize, class_num)
                 cos\theta given by P2SActivationLayer(input_feat)
          target: tensor (batchsize)
                 target[i] is the target class index of the i-th sample

        output
        ------
          loss: scaler
        """
        # target (batchsize, 1)
        target = target.long() #.view(-1, 1)
        
        # filling in the target
        # index (batchsize, class_num)
        with torch.no_grad():
            index = torch.zeros_like(input_score)
            # index[i][target[i][j]] = 1
            index.scatter_(1, target.data.view(-1, 1), 1)
    
        # MSE between \cos\theta and one-hot vectors
        loss = self.m_loss(input_score, index)

        return loss


class MSEP2SGRADLoss(torch_nn.Module):
    """
    Wrapper for P2SGradLoss to maintain compatibility with existing code.
    This loss function uses cosine angles as network output.
    """
    
    def __init__(self, **kwargs):
        super(MSEP2SGRADLoss, self).__init__()
        self.p2s_loss = P2SGradLoss()
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: model predictions (cosine angles)
            targets: ground truth labels (0 or 1)
        Returns:
            loss: computed P2SGrad loss value
        """
        if isinstance(predictions, dict):
            logits = predictions['logits']
        else:
            logits = predictions
            
        # Ensure targets are long for scatter operation
        targets = targets.long()
        
        return self.p2s_loss(logits, targets) 