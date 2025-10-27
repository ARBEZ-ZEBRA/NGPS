import numpy as np
import torch
import tinycudann as tcnn
import msgpack
import nerfacc
import math
from utilsS import inv_morton_naive, morton_naive

DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

class InstantNGPS(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Consts
        self.near = 0.6
        self.far = 2.0
        self.steps = 1024
        self.step_length = math.sqrt(3) / self.steps
        self.radius = 1.5
        self.render_step_length = self.radius * math.sqrt(3) / self.steps
        self.global_step = 0
        
        # Initialize models
        self.hash: torch.nn.Module = tcnn.NetworkWithInputEncoding(
            n_input_dims = 3,
            n_output_dims = 16,
            encoding_config = config["encoding"],
            network_config = config["network"]
        ).to("cuda")
        self.sh: torch.nn.Module = tcnn.Encoding(
            n_input_dims = 3,
            encoding_config = config["dir_encoding"],
            dtype = DTYPE
        ).to("cuda")
        self.mlp: torch.nn.Module = tcnn.Network(
            n_input_dims = 32,
            n_output_dims = 3,
            network_config = config["rgb_network"]
        ).to("cuda")
        grid_resolution = 128
        self.grid: torch.nn.Module = nerfacc.OccGridEstimator(
            roi_aabb = [0, 0, 0, 1, 1, 1],
            resolution = grid_resolution, levels = 1
        ).to("cuda")
        self.snapshot = None
        self.beta = torch.nn.Parameter(torch.tensor(1, dtype=DTYPE))
        self.gamma = torch.nn.Parameter(torch.tensor(1, dtype=DTYPE))
        self.cos = torch.nn.Parameter(torch.tensor(0.5, dtype=DTYPE))
        
        grid_1d = torch.abs(torch.randn(grid_resolution ** 3, dtype = DTYPE)) / 100
        
        grid_3d = torch.reshape((grid_1d > 0.01), [1, grid_resolution, grid_resolution, grid_resolution]).type(torch.bool)
        init_params_grid = {
            "resolution": torch.tensor([grid_resolution, grid_resolution, grid_resolution], dtype = torch.int32),
            "aabbs": torch.tensor([[0, 0, 0, 1, 1, 1]]),
            "occs": grid_1d,
            "binaries": grid_3d
        }
        self.grid.load_state_dict(init_params_grid)


    def load_snapshot(self, path: str):
        with open(path, 'rb') as f:
            unpacker = msgpack.Unpacker(f, raw = False, max_buffer_size = 0)
            snapshot = next(unpacker)
        self.snapshot = snapshot
        params_binary = torch.tensor(
            np.frombuffer(snapshot["snapshot"]["params_binary"], 
                        dtype = np.float16, offset = 0)#.astype(np.float32)
            , dtype = DTYPE)
        # Params for Hash Encoding Network
        ## Network Params Size: 32 * 64 + 64 * 16 = 3072
        params_hashnet = params_binary[:(32 * 64 + 64 * 16)]
        params_binary = params_binary[(32 * 64 + 64 * 16):]
        # Params for RGB Network
        ## Network Params Size: 32 * 64 + 64 * 64 + 64 * 16 = 7168
        params_rgbnet = params_binary[:(32 * 64 + 64 * 64 + 64 * 16)]
        params_binary = params_binary[(32 * 64 + 64 * 64 + 64 * 16):]
        # Params for Hash Encoding Grid
        params_hashgrid = params_binary
        # Params of Density Grid
        grid_raw = torch.tensor(
            np.frombuffer(
                snapshot["snapshot"]["density_grid_binary"], dtype=np.float16).astype(np.float32),
            dtype = DTYPE
            )
        grid = torch.zeros([128 * 128 * 128], dtype = DTYPE)

        x, y, z = inv_morton_naive(torch.arange(0, 128**3, 1))
        grid[x * 128 * 128 + y * 128 + z] = grid_raw
        grid_3d = torch.reshape(grid > 0.01, [1, 128, 128, 128]).type(torch.bool)
        
        params_hash = torch.cat([params_hashnet, params_hashgrid])
        self.hash.load_state_dict({"params": params_hash})
        self.mlp.load_state_dict({"params": params_rgbnet})
        self.grid.load_state_dict({
            "resolution": torch.tensor([128, 128, 128], dtype = torch.int32),
            "aabbs": torch.tensor([[0, 0, 0, 1, 1, 1]]),
            "occs": grid,
            "binaries": grid_3d
        })
    
    def save_snapshot(self, path: str, load_path: str | None = None):
        if load_path is not None:
            with open(load_path, 'rb') as f:
                unpacker = msgpack.Unpacker(f, raw = False, max_buffer_size = 0)
                snapshot = next(unpacker)
        # Parameters
        params_hash = self.hash.state_dict()["params"].clone().cpu().detach()
        params_rgbnet = self.mlp.state_dict()["params"].clone().cpu().detach()
        params_binary = torch.cat([
            params_hash[:(32 * 64 + 64 * 16)],
            params_rgbnet,
            params_hash[(32 * 64 + 64 * 16):]
        ]).numpy()
        snapshot["snapshot"]["params_binary"] = np.float16(params_binary).tobytes()
        # Density Grids
        density_grid: torch.Tensor = self.grid.state_dict()["occs"].clone().cpu().detach().type(torch.float16)
        grid_morton = torch.zeros(128 ** 3, dtype = torch.float16)
        indexs = torch.arange(0, 128**3, 1)
        grid_morton[morton_naive(indexs // (128 * 128), (indexs % (128 * 128)) // 128, indexs % 128)] = density_grid
        snapshot["snapshot"]["density_grid_binary"] = grid_morton.detach().numpy().tobytes()
        
        # HyperParameters
        snapshot['encoding']['log2_hashmap_size'] = self.config["encoding"]["log2_hashmap_size"]
        snapshot['encoding']['n_levels'] = self.config["encoding"]["n_levels"]
        snapshot['encoding']['n_features_per_level'] = self.config["encoding"]["n_features_per_level"]
        snapshot['encoding']['base_resolution'] = self.config["encoding"]["base_resolution"]
        with open(path, 'wb') as f:
            f.write(msgpack.packb(snapshot))

    def get_sdf(self, x: torch.Tensor):
        hash_features = self.hash(x)
        sdf = hash_features[..., 0]
        return sdf
    
    def get_beta(self, x: torch.Tensor):
        return self.beta.expand(x.shape[:-1])
    
    def get_gamma(self, x: torch.Tensor):
        return self.gamma.expand(x.shape[:-1])

    def get_alpha(self, x: torch.Tensor, dir: torch.Tensor):
        sdf = self.get_sdf(x).type(DTYPE)
        beta = self.get_beta(x).type(DTYPE)
        # gamma = self.get_gamma(x).type(DTYPE)

        normal = self.get_sdf_normal(x)
        true_cos = (dir * normal).sum(-1, keepdim=True)
        # true_cos = true_cos.squeeze(1)
        # true_cos = self.cos.expand(x.shape[:-1]).type(DTYPE)

        cos_anneal_ratio = min(1.0, self.global_step / 20000)
        iter_cos = (torch.nn.functional.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + torch.nn.functional.relu(-true_cos) * cos_anneal_ratio)
        iter_cos = iter_cos.squeeze(1)

        # alphas = (1. - torch.exp(-torch.exp(sdf) * self.step_length))
        # alphas = (1. - torch.exp(2*self.render_step_length*beta))*(1. - torch.exp(-torch.exp(-sdf * beta) * self.render_step_length))
        # alphas = (1. - torch.exp(gamma))*(1. - torch.exp(-torch.exp(-sdf * beta) * (iter_cos * self.render_step_length)))
        # alphas = (1. - torch.exp((-torch.exp(sdf) * beta) * (self.render_step_length * iter_cos)))
        # alphas = (1. - torch.exp((-torch.exp(sdf) * beta) * (self.render_step_length * (0.75 + 0.5*iter_cos))))
        alphas = (1. - torch.exp((-torch.exp(sdf) * beta) * (self.render_step_length)))
        # alphas = (1. - torch.exp(-2*self.render_step_length*iter_cos*beta)) * torch.sigmoid(-beta * sdf)
        # alphas = (1. - torch.exp(-beta)) * torch.sigmoid(-beta * (0.5 * sdf / (self.render_step_length) - 1))
        # alphas = 1. - torch.exp(-torch.exp(beta * (cos * dists - sdf)))
        # alphas = 1. - torch.exp(-torch.exp(beta * (-sdf)))
        # alphas = torch.sigmoid(-beta * sdf)

        # estimated_next_sdf = sdf - self.render_step_length * 0.5
        # estimated_prev_sdf = sdf + self.render_step_length * 0.5
        # prev_cdf = torch.sigmoid(estimated_prev_sdf * beta)
        # next_cdf = torch.sigmoid(estimated_next_sdf * beta)
        # p = prev_cdf - next_cdf
        # c = prev_cdf
        # alphas = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
        
        return alphas 

    def get_density(self, x: torch.Tensor):
        sdf = self.get_sdf(x).type(DTYPE)
        beta = self.get_beta(x).type(DTYPE)
        # gamma = self.get_gamma(x).type(DTYPE)

        density = torch.exp(sdf - 1) * self.step_length
        # density = (1. - torch.exp(2*self.render_step_length*beta))*(1. - torch.exp(-torch.exp(-sdf * beta) * self.render_step_length))
        # density = (1. - torch.exp(gamma))*(1. - torch.exp(-torch.exp(-sdf * beta) * (self.render_step_length)))
        # density = (1. - torch.exp(-torch.exp(sdf) * beta * self.render_step_length))
        # density = (1. - torch.exp(-2*self.render_step_length*beta)) * torch.sigmoid(-beta * sdf)
        # density = (1. - torch.exp(-beta)) * torch.sigmoid(-beta * (0.5 * sdf / (self.render_step_length) - 1))
        # density = 1. - torch.exp(-torch.exp(beta* (self.render_step_length - sdf)))
        # density = 1. - torch.exp(-torch.exp(beta * (-sdf)))
        # density = torch.sigmoid(-beta * sdf)
        
        return density

    def get_rgb(self, x: torch.Tensor, dir: torch.Tensor):
        hash_features = self.hash(x)
        sh_features = self.sh((dir + 1) / 2)     
        features = torch.concat([hash_features, sh_features], dim = -1)
        rgbs_raw = self.mlp(features)
        rgbs = torch.sigmoid(rgbs_raw)
        return rgbs
    
    def get_sdf_normal(self, x: torch.Tensor, eps=1e-3):
        requires_grad = x.requires_grad
        x.requires_grad_(True)
        
        sdf = self.get_sdf(x)
        
        normal = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=torch.ones_like(sdf),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
        x.requires_grad_(requires_grad)

        norm = torch.norm(normal, p=2, dim=-1, keepdim=True)
        normal = torch.where(norm > 1e-6, normal / norm, torch.zeros_like(normal))
    
        return normal

    def forward(self, position, direction):
        alphas = self.get_alpha(position, direction)
        rgbs = self.get_rgb(position, direction)
        normals = self.get_sdf_normal(position)
        density = self.get_density(position)

        self.global_step += 1
        return rgbs, alphas, normals, density