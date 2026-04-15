import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedRankingLoss(nn.Module):
    def __init__(self, rank_steps=None):
        super().__init__()
        self.rank_steps = rank_steps

    def forward(self, logits, costs, feasible_mask):
        current_mask = feasible_mask.bool().clone()
        max_steps = self.rank_steps or logits.size(1)
        total_loss = 0.0
        num_steps = 0

        for _ in range(max_steps):
            active_rows = current_mask.any(dim=1)
            if not active_rows.any():
                break

            step_logits = logits[active_rows].masked_fill(~current_mask[active_rows], -1e9)
            step_costs = costs[active_rows].masked_fill(~current_mask[active_rows], float("inf"))
            step_labels = torch.argmin(step_costs, dim=1)

            total_loss = total_loss + F.nll_loss(F.log_softmax(step_logits, dim=1), step_labels)
            num_steps += 1

            row_indices = active_rows.nonzero(as_tuple=False).squeeze(1)
            current_mask[row_indices, step_labels] = False

        if num_steps == 0:
            return logits.sum() * 0.0
        return total_loss / num_steps


class MaskedCrossEntropyLoss(nn.Module):
    def forward(self, logits, labels, feasible_mask):
        masked_logits = logits.masked_fill(~feasible_mask.bool(), -1e9)
        return F.nll_loss(F.log_softmax(masked_logits, dim=1), labels)
