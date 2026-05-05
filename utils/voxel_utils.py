import torch

class VoxelHash():
    """
    Simple Voxel Hash implementation, no consideration of hash collision
    """
    def __init__(self, resolution=None, buffer_size=int(1e7), device='cuda') -> None:
        
        self.resolution = resolution
        self.buffer_size = buffer_size

        self.dtype = torch.float32
        self.device = device

        # hash function
        self.primes = torch.tensor(
            [73856093, 19349669, 83492791], dtype=torch.int64, device=self.device)
       
        self.occ_list = torch.full([self.buffer_size], 0, dtype=torch.bool, device=self.device)


    def set_resolution(self, resolution):
        self.resolution = resolution


    def clear(self):
        self.occ_list = torch.full([self.buffer_size], 0, dtype=torch.bool, device=self.device)


    def update(self, points: torch.Tensor):

        round_points = torch.floor(points / self.resolution)

        # remove reptitive coordinates
        offset = round_points.min(dim=0,keepdim=True)[0]
        shift_points = round_points - offset
            
        v_size = round_points.max() + 1
        shift_idx = shift_points[:, 0] + shift_points[:, 1] * v_size + shift_points[:, 2] * v_size * v_size
        unique, index, counts = torch.unique(shift_idx, sorted=False ,return_inverse=True, return_counts=True)

        unique_points = torch.zeros_like(round_points[:len(counts), :], device=self.device)
            
        index.unsqueeze_(-1)
            
        unique_points.scatter_add_(-2, index.expand(round_points.shape), round_points)
        unique_points /= counts.unsqueeze(-1)

        # hash function
        keys = (unique_points.to(self.primes) * self.primes).sum(-1) % self.buffer_size

        update_mask = (self.occ_list[keys] == False)
        self.occ_list[keys[update_mask]] = True

            
    def get_valid_mask(self, query_points):
        
        round_queries = torch.floor(query_points / self.resolution).to(self.primes)
 
        query_keys = (round_queries * self.primes).sum(-1) % self.buffer_size

        valid_mask = (self.occ_list[query_keys] == False)

        return valid_mask
    
