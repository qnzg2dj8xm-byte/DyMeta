import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsExtractor(nn.Module):
    def __init__(self, concept_dim, hidden_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(concept_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        window = x[:, -5:, :]
        delta = x[:, -1, :] - window.mean(dim=1)
        var = window.var(dim=1, unbiased=False)
        energy = window.mean(dim=1)
        feat = torch.cat([delta, var, energy], dim=-1)
        return self.fc(feat)

class DualMechanismNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=4, future_steps=5):
        super().__init__()
        self.future_steps = future_steps
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)

        self.fc_states = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, num_classes))
            for _ in range(future_steps)
        ])

        self.dynamics = DynamicsExtractor(input_dim, hidden_dim=32)
        self.fc_trans = nn.Sequential(
            nn.Linear(hidden_dim + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, return_emb=False):
        out_seq, (hn, cn) = self.lstm(x)
        emb_final = hn[-1]
        logits_s = torch.stack([head(emb_final) for head in self.fc_states], dim=1)
        dyn_feat = self.dynamics(x)
        trans_input = torch.cat([emb_final.detach(), dyn_feat], dim=-1)
        logits_t = self.fc_trans(trans_input)

        if return_emb:
            return logits_s, logits_t, emb_final, out_seq
        return logits_s, logits_t

class PrototypeRegistry:
    def __init__(self, emb_dim=64, num_classes=4, momentum=0.9, device='cuda'):
        self.prototypes = torch.zeros(num_classes, emb_dim, device=device)
        self.momentum = momentum
        self.is_initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.num_classes = num_classes

    def update(self, features, labels):
        features = features.detach()
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.any():
                class_feat = features[mask].mean(dim=0)
                if not self.is_initialized[c]:
                    self.prototypes[c] = class_feat
                    self.is_initialized[c] = True
                else:
                    self.prototypes[c] = self.momentum * self.prototypes[c] + (1 - self.momentum) * class_feat
        self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=1)

    def get_prototypes(self):
        return self.prototypes