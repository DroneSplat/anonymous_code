import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.pose_utils import get_tensor_from_camera
from utils.camera_utils import visualizer
import cv2
import numpy as np
from tqdm import tqdm

def render_sets(
    pipe: PipelineParams,
    dataset: ModelParams,
    iteration: int,
    args,
):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        viewpoint_stack = scene.getTestCameras().copy()
        save_dir = os.path.join(scene.model_path, "render_test")
        makedirs(save_dir, exist_ok=True)
        for i, viewpoint_cam in tqdm(enumerate(viewpoint_stack)):
            pose = gaussians.get_RT_test(viewpoint_cam.uid)
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            image_name = viewpoint_cam.image_name + ".jpg"
            save_path = os.path.join(save_dir, image_name)
            torchvision.utils.save_image(image, save_path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    pp = PipelineParams(parser)

    model = ModelParams(parser, sentinel=True)

    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    render_sets(
        pp, 
        model.extract(args),
        args.iteration,
        args,
    )