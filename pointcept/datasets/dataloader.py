# pointcept/datasets/dataloader.py
import math
import torch
from torch.utils.data import DataLoader
#from pointcept.utils.misc import collate_fn

# 如果collate_fn不可用，定义一个简单的版本
try:
    from pointcept.utils.misc import collate_fn
except ImportError:
    def collate_fn(batch):
        """Simple collate function for point cloud data."""
        if isinstance(batch[0], dict):
            result = {}
            for key in batch[0].keys():
                if key == 'offset':
                    # Handle offset separately
                    offsets = [0]
                    for i, item in enumerate(batch):
                        if i == 0:
                            continue
                        offsets.append(offsets[-1] + len(batch[i-1]['coord']))
                    result[key] = torch.tensor(offsets)
                elif isinstance(batch[0][key], torch.Tensor):
                    # Concatenate tensors
                    result[key] = torch.cat([item[key] for item in batch], dim=0)
                elif hasattr(batch[0][key], '__array__'):
                    # Convert arrays to tensor and concatenate
                    result[key] = torch.cat([torch.from_numpy(item[key]) for item in batch], dim=0)
                else:
                    # Keep as list for other types
                    result[key] = [item[key] for item in batch]
            return result
        else:
            # Fallback to default collate
            return torch.utils.data.dataloader.default_collate(batch)

class InfiniteDataLoader:
    """Infinite data loader for continuous training."""

    def __init__(self, dataset, batch_size=1, num_workers=0, shuffle=False,
                 drop_last=False, pin_memory=True, persistent_workers=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0

        # Create initial data loader
        self._create_dataloader()

    def _create_dataloader(self):
        """Create a new data loader."""
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers
        )
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            # Recreate data loader when exhausted
            self._create_dataloader()
            data = next(self.iterator)
        return data

    def __len__(self):
        return len(self.dataloader)