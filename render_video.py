import torch
import splines
import splines.quaternion
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
import imageio
from scipy.spatial.transform import Rotation as R
import splines
import splines.quaternion
from scene.cameras import Camera


def kochanek_bartels_interpolation(keyframes, num_frames, tension=0.0, bias=0.0, continuity=0.0):
    positions = np.array([kf[0] for kf in keyframes])
    quaternions = np.array([kf[1] for kf in keyframes])

    position_spline = splines.KochanekBartels(
        positions, 
        tcb=(tension, bias, continuity),
        endconditions="natural"
    )

    orientation_spline = splines.quaternion.KochanekBartels(
        [splines.quaternion.UnitQuaternion.from_unit_xyzw(np.roll(q, shift=-1)) for q in quaternions],
        tcb=(tension, bias, continuity),
        endconditions="natural"
    )

    times = np.linspace(0, len(keyframes) - 1, num_frames)
    interpolated_positions = position_spline.evaluate(times)
    interpolated_orientations = orientation_spline.evaluate(times)
    interpolated_orientations = np.stack([np.array([quat.scalar, *quat.vector]) for quat in interpolated_orientations])

    return interpolated_positions, interpolated_orientations


def interpolate_camera_list(camera_list, n_frames, tension=0.0, bias=0.0, continuity=0.0):

    keyframes = []
    for camera in camera_list:
        position = camera.T
        rotation = R.from_matrix(camera.R).as_quat()  
        keyframes.append((position, rotation))

    print("len(key_frames):", len(keyframes))
    interpolated_positions, interpolated_orientations = kochanek_bartels_interpolation(keyframes, n_frames, tension, bias, continuity)

    FoVx = camera_list[0].FoVx
    FoVy = camera_list[0].FoVy
    new_camera_list = []
    for i in range(n_frames):
        pos = interpolated_positions[i]
        quat = interpolated_orientations[i]     

        R_matrix = R.from_quat(quat).as_matrix()

        original_camera = camera_list[i % len(camera_list)] 

        new_camera = Camera(
            colmap_id=original_camera.colmap_id,
            R=R_matrix,
            T=pos,
            FoVx=FoVx,
            FoVy=FoVy,
            image=original_camera.original_image, 
            gt_alpha_mask=original_camera.original_image * 0, 
            image_name=original_camera.image_name,
            uid=original_camera.uid,
            trans=original_camera.trans,
            scale=original_camera.scale,
            data_device=original_camera.data_device
        )
        
        new_camera_list.append(new_camera)
    print("len(new_camera_list):", len(new_camera_list))
    return new_camera_list

def images_to_video(image_folder, output_video_path, fps=30):
    """
    Convert images in a folder to a video.

    Args:
    - image_folder (str): The path to the folder containing the images.
    - output_video_path (str): The path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    """
    images = []

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            image_path = os.path.join(image_folder, filename)
            image = imageio.imread(image_path)
            images.append(image)

    imageio.mimwrite(output_video_path, images, fps=fps)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)

    print("num views: ", len(views))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
        rendering = render(
            view, gaussians, pipeline, background, camera_pose=camera_pose
        )["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )

def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    args,
):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_cameras = interpolate_camera_list(scene.getTestCameras(), args.n_views)
    # render interpolated views
    render_set(
        dataset.model_path,
        "interps",
        scene.loaded_iter,
        render_cameras,
        gaussians,
        pipeline,
        background,
    )

    image_folder = os.path.join(dataset.model_path, f'interps/ours_{args.iteration}/renders')
    output_video_file = os.path.join(dataset.model_path, f'interps.mp4')
    images_to_video(image_folder, output_video_file, fps=args.fps)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--n_views", default=600, type=int)
    parser.add_argument("--fps", default=30, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args,
    )
