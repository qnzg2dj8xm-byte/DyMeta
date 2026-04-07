import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Confidence Calibration
class TemperatureScaler(nn.Module):
    def __init__(self, init_temp=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits):
        temp = torch.clamp(self.temperature, min=1e-3)
        return logits / temp


# 2. Metacognitive Intervention Sys2
class MetacognitiveIntervention(nn.Module):
    def __init__(self, entropy_threshold=0.6, alpha_sys2=0.85):
        super().__init__()
        self.entropy_threshold = entropy_threshold
        self.alpha_sys2 = alpha_sys2  

    def compute_entropy(self, probs):
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
        return entropy

    def get_prototype_logits(self, embeddings, prototypes, temperature=0.1):
        """
        Compute the cosine similarity between the current frame features and each state prototype, 
        then convert them into a new logits distribution.
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        
        sim_matrix = torch.matmul(embeddings, prototypes.T) 
        return sim_matrix / temperature

    def forward(self, sys1_logits, embeddings, prototypes):
        """
        Execute the metacognitive correction loop.

        Returns:
            sys2_probs: Final corrected probability distribution
            sys2_trigger: Intervention trigger flag (1: Sys2 triggered, 0: not triggered)
            entropy: Original information entropy of Sys1
        """
        sys1_probs = F.softmax(sys1_logits, dim=1)
        
        entropy = self.compute_entropy(sys1_probs)
        
        sys2_trigger = (entropy > self.entropy_threshold).float().unsqueeze(1) # [batch_size, 1]
        
        if sys2_trigger.sum() > 0:
            proto_logits = self.get_prototype_logits(embeddings, prototypes)
            proto_probs = F.softmax(proto_logits, dim=1)
            
            calibrated_probs = (1 - self.alpha_sys2) * sys1_probs + self.alpha_sys2 * proto_probs
            
            sys2_probs = sys1_probs * (1 - sys2_trigger) + calibrated_probs * sys2_trigger
        else:
            sys2_probs = sys1_probs
            
        return sys2_probs, sys2_trigger.squeeze(1), entropy


# 3. XAI - Integrated Gradients
class IntegratedGradientsXAI:
    """
    Integrated Gradients algorithm.
    """
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