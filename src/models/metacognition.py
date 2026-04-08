import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =====================================================================
# 1. 动态置信度校准 (Temperature Scaling)
# =====================================================================
class TemperatureScaler(nn.Module):
    def __init__(self, init_temp=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits):
        temp = torch.clamp(self.temperature, min=1e-3)
        return logits / temp

    def fit(self, logits, labels, lr=0.01, max_iter=50):
        self.to(logits.device)
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss
            
        optimizer.step(eval_step)
        return self.temperature.item()


class MetacognitiveArbitrationModule:
    def __init__(self, t_dist=0.6, t_dyn=1.5, t_entropy=1.0, t_transition=0.5):
        self.t_dist = t_dist
        self.t_dyn = t_dyn
        self.t_entropy = t_entropy
        self.t_transition = t_transition

    def route(self, entropy_s, proto_dist, dynamics, transition_prob):
        action_defer = proto_dist > self.t_dist
        
        action_expand = (dynamics > self.t_dyn) & (transition_prob > self.t_transition) & (~action_defer)
        
        action_masking = (entropy_s > self.t_entropy) & (~action_defer) & (~action_expand)
        
        action_pass = (~action_defer) & (~action_expand) & (~action_masking)

        return {
            "defer": action_defer,
            "expand": action_expand,
            "mask": action_masking,
            "pass": action_pass
        }


class CalibratedMetacognitiveLoop(nn.Module):
    def __init__(self, base_model, anchor_dict_state, prototypes,
                 temperature=1.0, entropy_s_thresh=1.0, proto_dist_thresh=0.4,
                 mask_lr=1.0, max_iters=3, dyn_thresh=1.5):
        super().__init__()
        self.model = base_model
        self.anchor_dict_state = anchor_dict_state
        self.prototypes = prototypes
        self.temperature = temperature
        
        self.mask_lr = mask_lr
        self.max_iters = max_iters
        
        # 实例化 MAM 路由器
        self.arbitrator = MetacognitiveArbitrationModule(
            t_dist=proto_dist_thresh,
            t_dyn=dyn_thresh,
            t_entropy=entropy_s_thresh,
            t_transition=0.5 
        )

    def forward(self, x_seq, apply_intervention=True):
        device = x_seq.device
        batch_size, seq_len, num_concepts = x_seq.shape

        self.model.eval()
        with torch.no_grad():
            logits_s_all, logits_t, emb, _ = self.model(x_seq, return_emb=True)
            logits_s = logits_s_all[:, -1, :]
            
            probs_s = F.softmax(logits_s / self.temperature, dim=-1)
            entropy_s = -torch.sum(probs_s * torch.log(probs_s + 1e-9), dim=-1)
            probs_t = torch.sigmoid(logits_t).squeeze(-1)

            emb_norm = F.normalize(emb, p=2, dim=1)
            cos_sim = torch.matmul(emb_norm, self.prototypes.T)
            proto_dist = 1.0 - torch.max(cos_sim, dim=1)[0]

            delta_c = x_seq[:, -1, :] - x_seq[:, -5:, :].mean(dim=1)
            dynamics_score = torch.norm(delta_c, p=2, dim=-1)

        # 3. 呼叫 MAM 调度策略
        routes = self.arbitrator.route(entropy_s, proto_dist, dynamics_score, probs_t)
        
        final_probs_s = probs_s.clone()
        raw_mask_s = torch.zeros((batch_size, num_concepts), device=device)
        
        if apply_intervention:

            if routes["defer"].any():
                idx_defer = torch.where(routes["defer"])[0]
                final_probs_s[idx_defer] = 1.0 / final_probs_s.shape[-1]
                
            if routes["expand"].any():
                idx_expand = torch.where(routes["expand"])[0]
                smoothed_probs = F.softmax(logits_s[idx_expand] / (self.temperature * 2.0), dim=-1)
                final_probs_s[idx_expand] = smoothed_probs

            if routes["mask"].any():
                idx = torch.where(routes["mask"])[0]
                x_triggered = x_seq[idx]  
                
                self.model.train()
                with torch.enable_grad():
                    raw_mask_sub = torch.zeros((len(idx), num_concepts), device=device)
                    for _ in range(self.max_iters):
                        current_mask = torch.sigmoid(raw_mask_sub) * 2.0
                        masked_x = x_triggered * current_mask.unsqueeze(1)
                        masked_x.requires_grad_(True)

                        logits_sub_all, _, _, _ = self.model(masked_x, return_emb=True)
                        logits_sub = logits_sub_all[:, -1, :]
                        tentative_classes = torch.argmax(F.softmax(logits_sub, dim=-1), dim=-1)

                        # IG
                        ig_steps = 20
                        baseline = torch.zeros_like(masked_x)
                        alphas = torch.linspace(0, 1, steps=ig_steps, device=device).reshape(-1, 1, 1, 1)
                        scaled_inputs = baseline.unsqueeze(0) + alphas * (masked_x.unsqueeze(0) - baseline.unsqueeze(0))
                        scaled_inputs = scaled_inputs.reshape(ig_steps * len(idx), seq_len, num_concepts)
                        scaled_inputs.requires_grad_(True)

                        logits_ig_all, _, _, _ = self.model(scaled_inputs, return_emb=True)
                        logits_ig = logits_ig_all[:, -1, :]

                        target_classes_ig = tentative_classes.repeat(ig_steps)
                        indices_ig = torch.arange(ig_steps * len(idx))
                        scores_ig = logits_ig[indices_ig, target_classes_ig].sum()
                        grads_ig = torch.autograd.grad(scores_ig, scaled_inputs)[0]

                        rt_attribution = ((masked_x - baseline) * grads_ig.reshape(ig_steps, len(idx), seq_len, num_concepts).mean(dim=0)).abs().mean(dim=1)
                        rt_attribution = rt_attribution / (rt_attribution.max(dim=1, keepdim=True)[0] + 1e-9)

                        anchor_batch = torch.stack([self.anchor_dict_state[c.item()] for c in tentative_classes])
                        divergence = F.relu(rt_attribution - anchor_batch)
                        weight = (entropy_s[idx] / 1.386).unsqueeze(1)
                        penalty = divergence * weight
                        raw_mask_sub = raw_mask_sub - self.mask_lr * penalty

                    raw_mask_s[idx] = raw_mask_sub

                self.model.eval()
                with torch.no_grad():
                    final_mask_s_sub = torch.sigmoid(raw_mask_s) * 2.0
                    logits_s_final_all, _, _, _ = self.model(x_seq * final_mask_s_sub.unsqueeze(1), return_emb=True)
                    final_probs_s[idx] = F.softmax(logits_s_final_all[idx, -1, :] / self.temperature, dim=-1)

        final_mask_s = torch.sigmoid(raw_mask_s) * 2.0
        return final_probs_s, probs_t, dynamics_score, final_mask_s, routes



class IntegratedGradientsXAI:
    def __init__(self, model):
        self.model = model
        
    def generate_attributions(self, input_tensor, target_class, steps=50):
        self.model.eval()
        input_tensor.requires_grad_()
        
        baseline = torch.zeros_like(input_tensor)
        alphas = torch.linspace(0, 1, steps, device=input_tensor.device)
        interpolated = baseline + alphas.view(-1, 1, 1) * (input_tensor - baseline)
        interpolated.requires_grad_()
        
        preds = self.model(interpolated) 
        if isinstance(preds, tuple):
            logits_s = preds[0]  
        else:
            logits_s = preds
            
        if logits_s.dim() == 3:
            logits_s = logits_s[:, -1, :]
            
        target_scores = logits_s[:, target_class]
        
        gradients = torch.autograd.grad(outputs=target_scores, 
                                        inputs=interpolated,
                                        grad_outputs=torch.ones_like(target_scores),
                                        create_graph=False)[0]
        
        avg_gradients = gradients.mean(dim=0)
        attributions = (input_tensor[0] - baseline[0]) * avg_gradients
        
        return attributions.detach().cpu().numpy()