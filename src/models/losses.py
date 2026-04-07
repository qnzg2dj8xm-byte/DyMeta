import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()

class TemporalBehaviorContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, out_seq, emb_final, labels):
        device = out_seq.device
        batch_size, seq_len, hidden_dim = out_seq.shape

        loss_time = torch.tensor(0.0, device=device)
        loss_beh = torch.tensor(0.0, device=device)

        if self.alpha > 0 and seq_len > 1:
            z_t = F.normalize(out_seq[:, :-1, :].reshape(-1, hidden_dim), dim=1)
            z_t1 = F.normalize(out_seq[:, 1:, :].reshape(-1, hidden_dim), dim=1)

            sim_matrix_time = torch.matmul(z_t, z_t1.T) / self.temperature
            labels_time = torch.arange(z_t.size(0), device=device)
            loss_time = F.cross_entropy(sim_matrix_time, labels_time)

        if self.alpha < 1.0:
            z_beh = F.normalize(emb_final, dim=1)
            sim_matrix_beh = torch.matmul(z_beh, z_beh.T) / self.temperature

            labels_beh = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels_beh, labels_beh.T).float().to(device)

            logits_mask = torch.scatter(
                torch.ones_like(mask), 1,
                torch.arange(batch_size).view(-1, 1).to(device), 0
            )
            mask = mask * logits_mask

            logits_max, _ = torch.max(sim_matrix_beh, dim=1, keepdim=True)
            logits = sim_matrix_beh - logits_max.detach()

            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
            loss_beh = -mean_log_prob_pos.mean()

        return self.alpha * loss_time + (1 - self.alpha) * loss_beh

def supcon_loss(features, labels, temperature=0.1):
    device = features.device
    features = F.normalize(features, p=2, dim=1)
    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits_mask = torch.scatter(
        torch.ones_like(mask), 1,
        torch.arange(batch_size).view(-1, 1).to(device), 0
    )
    mask = mask * logits_mask
    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
    
    return -mean_log_prob_pos.mean()