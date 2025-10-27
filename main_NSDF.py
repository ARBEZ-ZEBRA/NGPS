import numpy as np
import torch
import json
import os
import math
import cv2 as cv
from tqdm import trange, tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import argparse
import open3d as o3d
from Dataset import NeRFSynthetic

from NGPS import InstantNGPS
from utilsS import Camera, render_image, render_image_o, render_image_d, render_image_n

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type = str, default = "mic")
parser.add_argument("--config", type = str, default = "base")
parser.add_argument("--max_steps", type = int, default = 25000)
parser.add_argument("--load_snapshot", "--load", type = str, default = "None")
parser.add_argument("--batch", "--batch_size", type = int, default = 8192)
parser.add_argument("--near_plane", "--near", type = float, default = 0.6)
parser.add_argument("--far_plane", "--far", type = float, default = 2.0)
parser.add_argument("--ray_marching_steps", type = int, default = 1024)

if (1):#__name__ == "__main__":
    torch.cuda.init()
    def print_memory_status():
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        print(torch.cuda.memory_summary())

    args = parser.parse_args()
    config_path = f"./configs/{args.config}.json"
    scene_name = args.scene
    max_steps = args.max_steps
    batch_size = args.batch
    near = args.near_plane
    far = args.far_plane
    ngp_steps = args.ray_marching_steps
    step_length = math.sqrt(3) / ngp_steps
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Evaluate Parameters
    with open(f"./data/nerf_synthetic/{scene_name}/transforms_test.json", "r") as f:
        meta = json.load(f)
    
    # Datasets
    dataset = NeRFSynthetic(f"./data/nerf_synthetic/{scene_name}")

    # Initialize models
    ngp: InstantNGPS = InstantNGPS(config).to("cuda")
    
    if args.load_snapshot != "None":
        ngp.load_snapshot(args.load_snapshot)
    weight_decay = (
        1e-5 if scene_name in ["materials", "ficus", "drums"] else 1e-6
    ) 

    optimizer = torch.optim.Adam(
        ngp.parameters(), lr=1e-2, eps=1e-15, weight_decay=weight_decay
    )

    # Train Utils
    grad_scaler = torch.amp.GradScaler('cuda', 2**10)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=100),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[ max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10,],
            gamma=0.33,
    ),]) 

    # Training
    ngp.train()
    ngp.grid.train()
    for step in trange(max_steps + 1):
        def occ_eval_fn(x):
            density = ngp.get_density(x)
            
            return density
        ngp.grid.update_every_n_steps(step = step, occ_eval_fn = occ_eval_fn, occ_thre = 1e-2)
        
        pixels, rays_o, rays_d = dataset.sample(batch_size)
        pixels = pixels.cuda()
        color = render_image(
            ngp, ngp.grid, rays_o, rays_d
        )
        loss = torch.nn.functional.smooth_l1_loss(color, pixels)
        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        optimizer.step()
        scheduler.step()
        
        # Eval
        if step == max_steps:
            dic = {"chair":"0000", "drums":"0001", "ficus":"0136", "hotdog":"0029", "lego":"0001", "materials":"0000", "mic":"0186", "ship":"0002"}
            psnr = [0 for i in range(10)]
            dpsnr = [0 for i in range(10)]
            for pic_num in [20*i for i in range(10)]:
                m_Camera_Angle_X = float(meta["camera_angle_x"])
                m_C2W = np.array(meta["frames"][pic_num]["transform_matrix"]).reshape(4, 4)
                camera = Camera((800, 800), m_Camera_Angle_X, m_C2W)

                ref_raw = cv.imread(f"./data/nerf_synthetic/{scene_name}/test/r_{pic_num}.png", cv.IMREAD_UNCHANGED) / 255.
                ref_raw = ref_raw[..., :3] * ref_raw[..., 3:]
                ref = np.array(ref_raw, dtype=np.float32)

                dep_raw = cv.imread(f"./data/nerf_synthetic/{scene_name}/test/r_{pic_num}_depth_{dic[scene_name]}.png", cv.IMREAD_UNCHANGED) / 255.
                dep_raw = dep_raw[..., :3] * dep_raw[..., 3:]
                dep = np.array(dep_raw, dtype=np.float32)
                dep = (dep) * 0.5 / (np.mean(dep) + 1e-6)
                dep = dep.clip(0, 1)

                total_color = np.zeros([800 * 800, 3], dtype = np.float32)
                total_depth = np.zeros([800 * 800, 3], dtype = np.float32)

                val_batch = 100 * 100
                for i in range(0, 800*800, val_batch):
                    rays_o_total = torch.tensor(camera.rays_o[i: i+val_batch], dtype = torch.float32)
                    rays_d_total = torch.tensor(camera.rays_d[i: i+val_batch], dtype = torch.float32)
                    color = render_image(ngp, ngp.grid, rays_o_total, rays_d_total,).cpu().detach().numpy()
                    depth = render_image_d(ngp, ngp.grid, rays_o_total, rays_d_total,).cpu().detach().numpy()
                
                    total_color[i: i+val_batch] = color
                    total_depth[i: i+val_batch] = depth
                
                    torch.cuda.empty_cache()

                image = np.clip(total_color[..., [2, 1, 0]].reshape(800, 800, 3), 0, 1)

                image_depth = total_depth[..., [2, 1, 0]].reshape(800, 800, 3)
                image_depth = (image_depth) / (np.max(image_depth) + 1e-6)
                image_depth = image_depth.clip(0, 1)

                psnr[pic_num//20] = compute_psnr(image, ref)
                dpsnr[pic_num//20] = compute_psnr(image_depth, dep)

                # if (pic_num == 0):
                #     cv.imwrite(f"./{scene_name}_depth.png", dep * 255.)
                #     cv.imwrite(f"./{scene_name}_depth_.png", image_depth * 255.)

                # tqdm.write(f"pic{pic_num}, PSNR = {round(psnr[pic_num].item(), 4)}, PSNR = {round(dpsnr[pic_num].item(), 4)}")

            PSNR = sum(psnr) / len(psnr)
            DPSNR = sum(dpsnr) / len(dpsnr)
            tqdm.write(f"PSNR = {round(PSNR.item(), 4)}, PSNR = {round(DPSNR.item(), 4)}")
        torch.cuda.empty_cache()
    
    # os.makedirs("Results", exist_ok = True)
    # os.makedirs(f"Results/Hash{config['encoding']['log2_hashmap_size']}", exist_ok = True)
    # ngp.save_snapshot(path = f"./Results/Hash{config['encoding']['log2_hashmap_size']}/{scene_name}.msgpack", load_path = "./snapshots/base.msgpack")

