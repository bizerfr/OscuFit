from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Display.SimpleGui import init_display

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.gp import gp_Dir
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps

from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.gp import gp_Pnt2d
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.TopAbs import TopAbs_REVERSED

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

import open3d as o3d
import numpy as np
from tqdm import tqdm

from PIL import Image, ImageDraw
from scipy.signal import lfilter
import fpsample

import os
import glob

np.random.seed(2025)

def load_step_file(path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    
    if status != IFSelect_RetDone:
        raise RuntimeError("Error: cannot read STEP file")
    
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape  # this is already a TopoDS_Shape

def get_face_area(face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return props.Mass()

def sample_surface(face, num_samples):
    surface = BRep_Tool.Surface(face)
    adaptor = BRepAdaptor_Surface(face)
    umin, umax = adaptor.FirstUParameter(), adaptor.LastUParameter()
    vmin, vmax = adaptor.FirstVParameter(), adaptor.LastVParameter()

    points, normals, max_curvatures, min_curvatures, max_dirs, min_dirs = [], [], [], [], [], []

    while len(points) < num_samples:

        u = np.random.uniform(umin, umax)
        v = np.random.uniform(vmin, vmax)

        # Check if (u, v) is inside the trimmed region
        classifier = BRepClass_FaceClassifier()
        classifier.Perform(face, gp_Pnt2d(u, v), 1e-6)
        if classifier.State() not in (TopAbs_IN, TopAbs_ON):
            continue

        prop = GeomLProp_SLProps(surface, u, v, 2, 1e-6)
        if not prop.IsNormalDefined() or not prop.IsCurvatureDefined():
            continue

        p = prop.Value()
        n = prop.Normal()
        k1 = prop.MaxCurvature()
        k2 = prop.MinCurvature()
        max_dir = gp_Dir()
        min_dir = gp_Dir()
        prop.CurvatureDirections(max_dir, min_dir)

        assert k1 >= k2
        if face.Orientation() == TopAbs_REVERSED:
            n.Reverse()
            max_dir.Reverse()
            min_dir.Reverse()
            k1, k2 = -k2, -k1
            max_dir, min_dir = min_dir, max_dir
            assert k1 >= k2

        points.append((p.X(), p.Y(), p.Z()))
        normals.append((n.X(), n.Y(), n.Z()))
        max_curvatures.append(k1)
        min_curvatures.append(k2)
        max_dirs.append((max_dir.X(), max_dir.Y(), max_dir.Z()))
        min_dirs.append((min_dir.X(), min_dir.Y(), min_dir.Z()))

    return points, normals, max_curvatures, min_curvatures, max_dirs, min_dirs


def farthest_point_sampling(points, k):
    points = np.array(points)
    selected = [np.random.randint(len(points))]
    distances = pairwise_distances(points, points[[selected[0]]]).reshape(-1)

    for _ in tqdm(range(1, k), desc="FPS"):
        next_index = np.argmax(distances)
        selected.append(next_index)
        dist_to_new = pairwise_distances(points, points[[next_index]]).reshape(-1)
        distances = np.minimum(distances, dist_to_new)

    return np.array(selected)

def add_brownian_noise_3d(points, noise_percent, alpha=0.01):
    N = points.shape[0]

    # Bounding box diameter
    max_extent = np.max(points, axis=0) - np.min(points, axis=0)
    diameter = np.linalg.norm(max_extent)

    # Generate white noise and apply IIR filter to get Brownian motion
    white = np.random.normal(0, 1, (N, 3))
    b, a = [alpha], [1, -(1 - alpha)]
    noise = np.stack([lfilter(b, a, white[:, i]) for i in range(3)], axis=1)

    # Normalize to desired noise level (standard deviation as % of diameter)
    noise /= (np.std(noise) + 1e-8)
    noise *= noise_percent * diameter

    return points + noise

def add_white_noise_3d(points, noise_percent):
    N = points.shape[0]

    # Bounding box diameter
    max_extent = np.max(points, axis=0) - np.min(points, axis=0)
    diameter = np.linalg.norm(max_extent)

    # Generate uncorrelated white noise
    noise = np.random.normal(0, 1, (N, 3))

    # Normalize to desired noise scale
    noise /= (np.std(noise) + 1e-8)
    noise *= noise_percent * diameter

    return points + noise

def sample_gradient_bbox_diagonal(points, sigma=0.3, target_n=100000):
    """
    Gradient sampling based on distance to bbox.min projected along the diagonal direction.
    - High sampling density near bbox.min
    - Sampling probability decays toward bbox.max
    """
    # 1. Compute bounding box diagonal
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    diag_vec = bbox_max - bbox_min
    diag_dir = diag_vec / (np.linalg.norm(diag_vec) + 1e-8)

    # 2. Project each point onto the diagonal vector
    proj = (points - bbox_min) @ diag_dir

    # 3. Normalize projection to [0, 1]
    proj_min, proj_max = proj.min(), proj.max()
    proj_norm = (proj - proj_min) / (proj_max - proj_min + 1e-8)

    # 4. Gaussian decay based on diagonal distance
    weights = np.exp(-(proj_norm ** 2) / (2 * sigma ** 2))
    weights += 1e-6
    weights /= weights.sum()

    # 5. Sample using weighted probability
    idx = np.random.choice(len(points), size=target_n, replace=False, p=weights)
    return idx

def sample_stripe_bbox_diagonal(points, freq=15, target_n=100000):
    """
    Stripe sampling along the diagonal of the bounding box.
    Produces alternating dense/sparse regions from bbox.min to bbox.max.
    """

    # 1. Get bounding box diagonal direction
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    diag_vec = bbox_max - bbox_min
    diag_dir = diag_vec / (np.linalg.norm(diag_vec) + 1e-8)

    # 2. Project each point onto that diagonal direction
    proj = (points - bbox_min) @ diag_dir  # scalar projection to diagonal

    proj_min, proj_max = proj.min(), proj.max()
    proj_norm = (proj - proj_min) / (proj_max - proj_min + 1e-8)

    weights = np.sin(freq * np.pi * proj_norm) ** 2 + 0.1
    weights += 1e-6
    weights /= weights.sum()

    idx = np.random.choice(len(points), size=target_n, replace=False, p=weights)
    return idx


parent_root = "/home/ntu/Dataset/ABC/step_v00/abc_0000_step_v00"
root_dir = "/home/ntu/Dataset/ABC-Diff"
root_dir_pwn = "/home/ntu/Dataset/ABC-Diff-PWN"


subdirs = sorted([os.path.join(parent_root, d) for d in os.listdir(parent_root)
                  if os.path.isdir(os.path.join(parent_root, d))])

for step_file_parent_root in tqdm(subdirs, desc="Processing Directories"):
    try:
        base_name = os.path.basename(os.path.normpath(step_file_parent_root))
        step_file_path = glob.glob(os.path.join(step_file_parent_root, "*.step"))[0]
        shape = load_step_file(step_file_path)

        target_total = 100000
        #target_total = 5000
        oversample_factor = 5
        oversample_count = target_total * oversample_factor

        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)

        # Estimate faces and distribute samples
        faces = []
        while face_explorer.More():
            face = face_explorer.Current()
            faces.append(face)
            face_explorer.Next()

        total_area = sum(get_face_area(f) for f in faces)
        samples_per_faces = [
            int((get_face_area(f) / total_area) * oversample_count) for f in faces]

        points, normals, max_curvatures, min_curvatures , max_dirs, min_dirs = [], [], [], [], [], []

        for fi, face in tqdm(enumerate(faces), total=len(faces), desc="Sampling faces"):
            points_, normals_, max_curvatures_, min_curvatures_, max_dirs_, min_dirs_ = sample_surface(face, samples_per_faces[fi])
            points.extend(points_)
            normals.extend(normals_)
            max_curvatures.extend(max_curvatures_)
            min_curvatures.extend(min_curvatures_)
            max_dirs.extend(max_dirs_)
            min_dirs.extend(min_dirs_)

        # Convert to numpy arrays
        points = np.array(points)

        normals = np.array(normals)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        normals = normals / norms

        max_curvatures = np.array(max_curvatures)
        min_curvatures = np.array(min_curvatures)
        max_dirs = np.array(max_dirs)
        min_dirs = np.array(min_dirs)

        idx_gradient = sample_gradient_bbox_diagonal(points, target_n=target_total)
        points_gradient = points[idx_gradient]
        normals_gradient = normals[idx_gradient]
        max_curvatures_gradient = max_curvatures[idx_gradient]
        min_curvatures_gradient = min_curvatures[idx_gradient]
        max_dirs_gradient = max_dirs[idx_gradient]
        min_dirs_gradient = min_dirs[idx_gradient]
        pwns_gradient = np.concatenate([points_gradient, normals_gradient], axis=1)
        print("Applying farthest point sampling to obtain uniform 5k points...")
        fps_idx_5k_gradient = fpsample.bucket_fps_kdtree_sampling(points_gradient, 5000)


        idx_stripe = sample_stripe_bbox_diagonal(points, target_n=target_total)
        points_stripe = points[idx_stripe]
        normals_stripe = normals[idx_stripe]
        max_curvatures_stripe = max_curvatures[idx_stripe]
        min_curvatures_stripe = min_curvatures[idx_stripe]
        max_dirs_stripe = max_dirs[idx_stripe]
        min_dirs_stripe = min_dirs[idx_stripe]
        pwns_stripe = np.concatenate([points_stripe, normals_stripe], axis=1)
        print("Applying farthest point sampling to obtain uniform 5k points...")
        fps_idx_5k_stripe = fpsample.bucket_fps_kdtree_sampling(points_stripe, 5000)

        # Farthest Point Sampling to get uniform 100k points
        print("Applying farthest point sampling to obtain uniform 100k points...")
        fps_idx = fpsample.bucket_fps_kdtree_sampling(points, target_total)

        points = points[fps_idx]
        normals = normals[fps_idx]
        max_curvatures = max_curvatures[fps_idx]
        min_curvatures = min_curvatures[fps_idx]
        max_dirs = max_dirs[fps_idx]
        min_dirs = min_dirs[fps_idx]
        pwns = np.concatenate([points, normals], axis=1)

        print("Applying farthest point sampling to obtain uniform 5k points...")
        fps_idx_5k = fpsample.bucket_fps_kdtree_sampling(points, 5000)

        points_noise_low = add_white_noise_3d(points, noise_percent=0.12/100)
        points_noise_middle = add_white_noise_3d(points, noise_percent=0.6/100)
        points_noise_high = add_white_noise_3d(points, noise_percent=1.2/100)

        pwns_noise_low = np.concatenate([points_noise_low, normals], axis=1)
        pwns_noise_middle = np.concatenate([points_noise_middle, normals], axis=1)
        pwns_noise_high = np.concatenate([points_noise_high, normals], axis=1)

        # save 
        root_dir = "/home/ntu/Dataset/ABC-Diff-v00"
        root_dir_pwn = "/home/ntu/Dataset/ABC-Diff-v00-PWN"
        np.savetxt(os.path.join(root_dir, f"{base_name}.xyz"), points, fmt="%.17g")
        np.savetxt(os.path.join(root_dir, f"{base_name}.normals"), normals, fmt="%.17g")
        np.savetxt(os.path.join(root_dir, f"{base_name}.curv"), np.stack([max_curvatures, min_curvatures], axis=1), fmt='%.17g')
        np.savetxt(os.path.join(root_dir, f"{base_name}.dir"), np.concatenate([max_dirs, min_dirs], axis=1), fmt='%.17g')
        np.savetxt(os.path.join(root_dir, f"{base_name}.pidx"), fps_idx_5k.reshape(-1, 1), fmt='%d')
        np.savetxt(os.path.join(root_dir_pwn, f"{base_name}.xyz"), pwns, fmt="%.17g")

        np.savetxt(os.path.join(root_dir, f"{base_name}_low_noise.xyz"), points_noise_low, fmt="%.17g")
        np.savetxt(os.path.join(root_dir, f"{base_name}_low_noise.normals"), normals, fmt="%.17g")
        np.savetxt(os.path.join(root_dir, f"{base_name}_low_noise.curv"), np.stack([max_curvatures, min_curvatures], axis=1), fmt='%.17g')
        np.savetxt(os.path.join(root_dir, f"{base_name}_low_noise.dir"), np.concatenate([max_dirs, min_dirs], axis=1), fmt='%.17g')
        np.savetxt(os.path.join(root_dir, f"{base_name}_low_noise.pidx"), fps_idx_5k.reshape(-1, 1), fmt='%d')
        np.savetxt(os.path.join(root_dir_pwn, f"{base_name}_low_noise.xyz"), pwns_noise_low, fmt="%.17g")

        np.savetxt(os.path.join(root_dir, f"{base_name}_middle_noise.xyz"), points_noise_middle, fmt="%.17g")
        np.savetxt(os.path.join(root_dir, f"{base_name}_middle_noise.normals"), normals, fmt="%.17g")
        np.savetxt(os.path.join(root_dir, f"{base_name}_middle_noise.curv"), np.stack([max_curvatures, min_curvatures], axis=1), fmt='%.17g')
        np.savetxt(os.path.join(root_dir, f"{base_name}_middle_noise.dir"), np.concatenate([max_dirs, min_dirs], axis=1), fmt='%.17g')
        np.savetxt(os.path.join(root_dir, f"{base_name}_middle_noise.pidx"), fps_idx_5k.reshape(-1, 1), fmt='%d')
        np.savetxt(os.path.join(root_dir_pwn, f"{base_name}_middle_noise.xyz"), pwns_noise_middle, fmt="%.17g")

        np.savetxt(os.path.join(root_dir, f"{base_name}_high_noise.xyz"), points_noise_high, fmt="%.17g")
        np.savetxt(os.path.join(root_dir, f"{base_name}_high_noise.normals"), normals, fmt="%.17g")
        np.savetxt(os.path.join(root_dir, f"{base_name}_high_noise.curv"), np.stack([max_curvatures, min_curvatures], axis=1), fmt='%.17g')
        np.savetxt(os.path.join(root_dir, f"{base_name}_high_noise.dir"), np.concatenate([max_dirs, min_dirs], axis=1), fmt='%.17g')
        np.savetxt(os.path.join(root_dir, f"{base_name}_high_noise.pidx"), fps_idx_5k.reshape(-1, 1), fmt='%d')
        np.savetxt(os.path.join(root_dir_pwn, f"{base_name}_high_noise.xyz"), pwns_noise_high, fmt="%.17g")

        np.savetxt(os.path.join(root_dir, f"{base_name}_gradient.xyz"), points_gradient, fmt="%.17g")
        np.savetxt(os.path.join(root_dir, f"{base_name}_gradient.normals"), normals_gradient, fmt="%.17g")
        np.savetxt(os.path.join(root_dir, f"{base_name}_gradient.curv"), np.stack([max_curvatures_gradient, min_curvatures_gradient], axis=1), fmt='%.17g')
        np.savetxt(os.path.join(root_dir, f"{base_name}_gradient.dir"), np.concatenate([max_dirs_gradient, min_dirs_gradient], axis=1), fmt='%.17g')
        np.savetxt(os.path.join(root_dir, f"{base_name}_gradient.pidx"), fps_idx_5k_gradient.reshape(-1, 1), fmt='%d')
        np.savetxt(os.path.join(root_dir_pwn, f"{base_name}_gradient.xyz"), pwns_gradient, fmt="%.17g")


        np.savetxt(os.path.join(root_dir, f"{base_name}_stripe.xyz"), points_stripe, fmt="%.17g")
        np.savetxt(os.path.join(root_dir, f"{base_name}_stripe.normals"), normals_stripe, fmt="%.17g")
        np.savetxt(os.path.join(root_dir, f"{base_name}_stripe.curv"), np.stack([max_curvatures_stripe, min_curvatures_stripe], axis=1), fmt='%.17g')
        np.savetxt(os.path.join(root_dir, f"{base_name}_stripe.dir"), np.concatenate([max_dirs_stripe, min_dirs_stripe], axis=1), fmt='%.17g')
        np.savetxt(os.path.join(root_dir, f"{base_name}_stripe.pidx"), fps_idx_5k_stripe.reshape(-1, 1), fmt='%d')
        np.savetxt(os.path.join(root_dir_pwn, f"{base_name}_stripe.xyz"), pwns_stripe, fmt="%.17g")

    except Exception as e:
        print(f"Failed processing {step_file_parent_root}: {e}")