import os
import numpy as np
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
import uuid
from scene import Scene, GaussianModel
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.pose_utils import get_camera_from_tensor
import matplotlib.pyplot as plt
import json
from PIL import Image
from scipy.ndimage import label, center_of_mass
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



def load_mask(mask_path):
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask) / 255.0
    return torch.tensor(mask, dtype=torch.float32).cuda()


def load_label_maps_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        label_maps = json.load(json_file)
    for key in label_maps:
        label_maps[key] = np.array(label_maps[key], dtype=np.uint16)
    return label_maps

def save_pose(path, quat_pose, train_cams, llffhold=2):
    output_poses = []
    index_colmap = [cam.colmap_id for cam in train_cams]

    if len(set(index_colmap)) == 1:
        index_colmap = list(range(1, len(train_cams)))
    
    for cam_id in index_colmap:
        ind = index_colmap.index(cam_id)
        w2c = get_camera_from_tensor(quat_pose[ind])
        output_poses.append(w2c)
        
    colmap_poses = torch.stack(output_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)

def compute_instance_loss_sum(combined_loss, label_map):

    instance_ids = torch.unique(label_map)
    instance_loss_sum = {}

    for instance_id in instance_ids:
        if instance_id.item() == 0: 
            continue
        mask = (label_map == instance_id).float()
        instance_area = mask.sum()  
        instance_loss = (combined_loss * mask).sum().item() / instance_area.item()
        instance_loss_sum[instance_id.item()] = instance_loss

    return instance_loss_sum

def compute_combined_loss(l1_loss_val, ssim_val):
    Ll1_norm = normalize_to_01(l1_loss_val.mean(dim=0)).detach()
    ssim_value_norm = normalize_to_01(ssim_val).detach()

    combined_loss = (4 * Ll1_norm + ssim_value_norm) / 5.0
    return combined_loss, Ll1_norm, ssim_value_norm

def compute_instance_losses(combined_loss, label_map, preset_instance_threshold, iteration, image_name, args):
    instance_loss_sum = compute_instance_loss_sum(combined_loss, label_map)
    heatmap = torch.zeros_like(label_map, dtype=torch.float32)

    for instance_id, loss_sum in instance_loss_sum.items():
        if instance_id == 0:
            continue
        mask = (label_map == instance_id).float()
        heatmap += mask * loss_sum

    heatmap_norm = normalize_to_01(heatmap)

    mean_value = heatmap_norm.mean()
    std_value = heatmap_norm.std()
    instance_threshold = mean_value + std_value + args.threshold_local * std_value * (args.iterations - iteration) / args.iterations

    heatmap_norm = torch.tensor(heatmap_norm, dtype=torch.float32).cuda()
    heatmap_binary = (heatmap_norm < instance_threshold).float()
    
    return heatmap_norm, heatmap_binary

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def normalize_to_01(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    return (tensor - tensor_min) / (tensor_max - tensor_min)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, opt=args, shuffle=True)                                                                      
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    train_cams_init = scene.getTrainCameras().copy()
    test_cams_init = scene.getTestCameras().copy()
    save_pose_path = os.path.join(scene.model_path, "pose")
    os.makedirs(save_pose_path, exist_ok=True)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if args.use_masks:
        source_path = args.source_path
        masks_path = os.path.join(source_path, "masks")
        mask_json_path = os.path.join(masks_path, "masks.json")
        label_maps = load_label_maps_from_json(mask_json_path)
    
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        pose = gaussians.get_RT(viewpoint_cam.uid)


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        if args.use_masks:
            # load mask
            mask_name = viewpoint_cam.image_name + ".jpg"
            mask_img = Image.open(os.path.join(masks_path, mask_name))
            mask_img = np.array(mask_img)
            img_name = viewpoint_cam.image_name + ".jpg"
            label_map_for_image = label_maps.get(img_name)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = 1.0 - ssim(image, gt_image)
        
        combined_loss, Ll1_norm, ssim_value_norm = compute_combined_loss(Ll1, ssim_value)

        if args.use_masks and label_map_for_image is not None:
            label_map_for_image = torch.tensor(label_map_for_image).cuda()
            instance_threshold = args.preset_instance_threshold
            image_name = viewpoint_cam.image_name
            heatmap_norm, heatmap_binary = compute_instance_losses(combined_loss, label_map_for_image, instance_threshold, iteration, image_name, args)

        if args.use_masks and iteration > args.mask_start_iter:
                loss = (1.0 - opt.lambda_dssim) * (Ll1 * heatmap_binary).mean() + opt.lambda_dssim * (ssim_value * heatmap_binary).mean()
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1.mean() + opt.lambda_dssim * ssim_value.mean()
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            
            densify_grad_threshold = opt.densify_grad_threshold
            if args.schedule_densify_grad_threshold:
                densify_grad_threshold = densify_grad_threshold + (0.001 - densify_grad_threshold) * (iteration / opt.iterations)

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1.mean(), loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_pose(save_pose_path + f"/pose_{iteration}.npy", gaussians.test_P, test_cams_init)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name']=="train":
                        pose = scene.gaussians.get_RT(viewpoint.uid)
                    else:
                        pose = scene.gaussians.get_RT_test(viewpoint.uid)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, camera_pose=pose)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 1000, 2000, 7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--preset_instance_threshold", type=float, default=0.4)
    parser.add_argument("--threshold_local", type=float, default=0.4)
    parser.add_argument("--use_masks", action="store_true")
    parser.add_argument("--mask_start_iter", type=int, default=500)
    parser.add_argument("--sam2_ckpt", type=str, default="checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--sam2_cfg", type=str, default="sam2_hiera_l.yaml")
    parser.add_argument("--schedule_densify_grad_threshold", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    os.makedirs(args.model_path, exist_ok=True)
    
    print("Optimizing " + args.model_path)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    print("\nTraining complete.")