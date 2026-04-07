import torch
import torch.nn as nn

class DynaMetaTrainer:
    def __init__(self, model, dataloader, optimizer, criterion_tbc, criterion_trans, proto_registry, device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion_tbc = criterion_tbc     
        self.criterion_trans = criterion_trans  
        self.proto_registry = proto_registry    
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_tbc_loss = 0.0
        total_trans_loss = 0.0
        
        for batch_X, batch_Y_fut, batch_Y_curr, batch_C in self.dataloader:
            batch_X = batch_X.to(self.device)
            batch_Y_fut = batch_Y_fut.to(self.device)
            batch_Y_curr = batch_Y_curr.to(self.device)
            batch_C = batch_C.to(self.device)

            self.optimizer.zero_grad()

            logits_s, logits_t, emb_final, out_seq = self.model(batch_X, return_emb=True)
            
            loss_states = 0.0
            for step in range(logits_s.size(1)):
                loss_states += nn.CrossEntropyLoss()(logits_s[:, step, :], batch_Y_fut)
            loss_states /= logits_s.size(1)

            loss_tbc = self.criterion_tbc(out_seq, emb_final, batch_Y_curr)

            is_trans = (batch_Y_fut != batch_Y_curr).float().unsqueeze(1)
            loss_trans = self.criterion_trans(logits_t, is_trans)

            loss = loss_states + 5.0 * loss_trans + 0.1 * loss_tbc
            
            loss.backward()
            self.optimizer.step()

            self.proto_registry.update(emb_final, batch_Y_curr)

            total_loss += loss.item()
            total_tbc_loss += loss_tbc.item()
            total_trans_loss += loss_trans.item()

        num_batches = len(self.dataloader)
        return {
            "loss": total_loss / num_batches,
            "loss_tbc": total_tbc_loss / num_batches,
            "loss_trans": total_trans_loss / num_batches
        }