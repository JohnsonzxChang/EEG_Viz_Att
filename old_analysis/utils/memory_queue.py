"""
FIFO Embedding Queue (MoCo-style)
==================================
Ring buffer storing recent (z, y) pairs from training batches.
Augments Circle Loss effective batch from batch_size to batch_size + queue_size
without additional GPU memory cost for forward/backward.

Queue entries are always detached (stop-gradient) — essential because they
are from different optimizer steps and their gradients would be stale.

Usage:
    queue = EmbeddingQueue(queue_size=512, embed_dim=128, num_classes=79, device='cuda')

    # In training loop:
    z = F.normalize(feat, dim=-1)
    z_aug, y_aug = queue.augmented_batch(z, batch_y)  # expand for Circle Loss
    l_circle = circle_loss_fn(z_aug, y_aug)
    queue.enqueue(z, batch_y)                          # update queue after loss
"""

import torch


class EmbeddingQueue:
    """Ring-buffer queue for (embedding, label) pairs.

    Args:
        queue_size:   maximum stored embeddings. Default 512.
        embed_dim:    embedding dimension. Default 128.
        num_classes:  label classes. Default 79.
        device:       torch device.
    """

    def __init__(self, queue_size: int = 512, embed_dim: int = 128,
                 num_classes: int = 79, device='cpu'):
        self.queue_size = queue_size
        self.ptr = 0
        self.full = False

        self.z_queue = torch.zeros(queue_size, embed_dim, device=device)
        self.y_queue = torch.zeros(queue_size, num_classes, device=device)

    @torch.no_grad()
    def enqueue(self, z: torch.Tensor, y: torch.Tensor):
        """Add batch to queue with wrap-around. Always detaches."""
        z = z.detach()
        y = y.detach()
        B = z.size(0)

        if B >= self.queue_size:
            self.z_queue = z[-self.queue_size:].clone()
            self.y_queue = y[-self.queue_size:].clone()
            self.ptr = 0
            self.full = True
            return

        if not self.full:
            # Standard FIFO fill if not full yet
            end = self.ptr + B
            if end <= self.queue_size:
                self.z_queue[self.ptr:end] = z
                self.y_queue[self.ptr:end] = y
            else:
                first = self.queue_size - self.ptr
                self.z_queue[self.ptr:] = z[:first]
                self.y_queue[self.ptr:] = y[:first]
                self.z_queue[:B - first] = z[first:]
                self.y_queue[:B - first] = y[first:]
            
            self.ptr = end % self.queue_size
            if end >= self.queue_size:
                self.full = True
        else:
            # Hard Negative Mining Eviction
            with torch.no_grad():
                # Compute redundancy: sum of similarities to other elements in queue
                sim_matrix = self.z_queue @ self.z_queue.T
                # Ignore self-similarity (the diagonal)
                sim_matrix.fill_diagonal_(0.0)
                redundancy_scores = sim_matrix.mean(dim=1)
                
                # Find the B most redundant indices to evict
                _, evict_indices = torch.topk(redundancy_scores, k=B, largest=True)
                
                # Replace them
                self.z_queue[evict_indices] = z
                self.y_queue[evict_indices] = y

    def get(self):
        """Return current queue contents (detached). None if empty."""
        if not self.full and self.ptr == 0:
            return None, None
        if self.full:
            return self.z_queue.clone(), self.y_queue.clone()
        return self.z_queue[:self.ptr].clone(), self.y_queue[:self.ptr].clone()

    def augmented_batch(self, z: torch.Tensor, y: torch.Tensor):
        """Concatenate current batch with queue contents for expanded pair mining."""
        q_z, q_y = self.get()
        if q_z is None:
            return z, y
        return torch.cat([z, q_z], dim=0), torch.cat([y, q_y], dim=0)
