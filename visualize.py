import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

def visualize_quad(u, query=None, patch=None):
    N = 100  
    bbox_min = patch.min(axis=0) - 1.0
    bbox_max = patch.max(axis=0) + 1.0
    extent = bbox_max - bbox_min

    voxel_size = extent.max() / (N - 1)

    nx = int(np.ceil(extent[0] / voxel_size)) + 1
    ny = int(np.ceil(extent[1] / voxel_size)) + 1
    nz = int(np.ceil(extent[2] / voxel_size)) + 1

    x_vals = np.linspace(bbox_min[0], bbox_min[0] + voxel_size * (nx - 1), nx)
    y_vals = np.linspace(bbox_min[1], bbox_min[1] + voxel_size * (ny - 1), ny)
    z_vals = np.linspace(bbox_min[2], bbox_min[2] + voxel_size * (nz - 1), nz)

    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

    phi = np.stack([X**2, Y**2, Z**2, X*Y, X*Z, Y*Z, X, Y, Z], axis=-1) 
    fval = phi @ u

    spacing = np.array([x_vals[1] - x_vals[0], y_vals[1] - y_vals[0], z_vals[1] - z_vals[0]])
    origin = np.array([x_vals[0], y_vals[0], z_vals[0]])
    verts, faces, normals, _ = measure.marching_cubes(fval, level=0.0,
        spacing=spacing)
    verts_phys = verts  + origin
    
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # mesh = Poly3DCollection(verts_phys[faces], alpha=0.7)
    # mesh.set_facecolor('cyan')
    # ax.set_box_aspect((1,1,1))
    # ax.add_collection3d(mesh)
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()

    mesh = trimesh.Trimesh(
        vertices=verts_phys,
        faces=faces,
        vertex_normals=normals)
    mesh.visual.face_colors = [255, 0, 0, 80]

    edges = mesh.edges_unique  
    segments = np.stack([mesh.vertices[edges[:, 0]], mesh.vertices[edges[:, 1]]], axis=1)
    wire = trimesh.load_path(segments.reshape(-1, 2, 3))  
    wire.colors = np.tile(np.array([[255, 0, 0, 80]]), (len(wire.entities), 1)) 

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name='mesh')
    scene.add_geometry(wire, geom_name='wire')
    
    
    if patch is not None:
        patch = trimesh.points.PointCloud(patch, colors=[0, 0, 0, 255])
        scene.add_geometry(patch, geom_name='patch')

    # if query is not None:
    #     query = np.expand_dims(query, axis=0)
    #     query = trimesh.points.PointCloud(query, colors=[255, 0, 0, 255])
    #     scene.add_geometry(query, geom_name='query')
    if query is not None:
        query = np.asarray(query).reshape(1, 3)  # ensure shape (1, 3)
        s = trimesh.creation.uv_sphere(radius=0.04)  # adjust size here
        s.apply_translation(query[0])
        s.visual.vertex_colors = [255, 0, 0, 255]  # solid red
        scene.add_geometry(s, geom_name='query')
    
    scene.show()


if __name__ == '__main__':
    u = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    query = np.array([0.0, 0.0, 0.0])
    patch = np.array([
        [0.5, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.0, 1.0, 0.0],])

    visualize_quad(u, query, patch)
    