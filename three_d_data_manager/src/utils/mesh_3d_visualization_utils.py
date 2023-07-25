import os
import subprocess

import torchvision
from PIL import Image
import torch
from tqdm import tqdm
from xvfbwrapper import Xvfb
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from PIL import Image
import plotly

from .mesh_utils import MeshContainer


# The following function is based on code from https://github.com/omriefroni/dfaust_allign/visualization/vizualization_functions.py
def visualize_grid_of_lbo(verts ,faces:np.ndarray ,eigenvectors:np.ndarray ,dirpath:str, max_lbos:int=None  , mesh_or_pc:str='mesh', prefix:str='', write_html:bool=True, save_image:bool=True, save_plotly_json:bool=True):
    imgs = []
    paths = {
        "jsons":[],
        "imgs":[],
        "htmls":[]
    }
    max_lbos = max_lbos or 14
    
    for eigvec_idx in range(min(max_lbos, eigenvectors.shape[1])):
        save_path = os.path.join(dirpath, f'{prefix}_{eigvec_idx}')
        fig:go.Figure = MeshVisualizer().visualize_scalar_vector_on_shape( 
            MeshContainer(verts, faces),
            eigenvectors[: ,eigvec_idx],
            mesh_or_pc=mesh_or_pc,
        )

        if write_html:
            fig.write_html(save_path + ".html") 
            paths["htmls"].append(save_path + ".html")
        if save_image:
            fig.write_image(save_path + ".png")
            paths["imgs"].append(save_path + ".png")
        if save_plotly_json:
            with open(save_path + ".json", 'w') as outfile:
                outfile.write(plotly.io.to_json(fig, pretty=True))
            paths["jsons"].append(save_path + ".json")

        imgs.append(torchvision.transforms.functional.to_tensor(Image.open(save_path + ".png")))

    torchvision.utils.save_image(torch.stack(imgs), nrow=2, fp=os.path.join(dirpath, f'{prefix}_lbo_grid.png'))

    paths["lbo_grid"] = save_path + '_lbo_grid.png'
    return paths

# The following function is taken from https://github.com/omriefroni/dfaust_allign/visualization/mesh_visualizer.py
def check_xvfb():

    comm = 'ps -ef | grep Xvfb | grep -v grep | cat'
    output = subprocess.check_output(comm, shell=True)

    if len(output) == 0:
        return False
    else:
        return True

# The following class is based on code from https://github.com/omriefroni/dfaust_allign/visualization/mesh_visualizer.py
class MeshVisualizer:
    """Visualization class for meshs."""

    def __init__(self,  display_up=False): #dataset="FAUST",
        """
        The mesh visualization utility class

        Args:
            # dataset (str, optional): Name of the dataset, e.g. will effect view angle . Defaults to 'FAUST'.
        """
        # self.dataset = dataset
        self.scale = 0.2
        # self.set_view_properties_by_dataset()
        if not check_xvfb():
            vdisplay = Xvfb()
            vdisplay.start()
            self.display_eng = vdisplay
        else:
            self.display_eng = None

    @staticmethod
    def set_fig_settings(scene, fig, camera, width=600, height=600, scale=False):
        fig.layout.update(autosize=False, width=width, height=height, margin=dict(l=0, r=0, t=0, b=0))

        scene.camera = camera
        scene.aspectmode = "data"
        scene.xaxis.visible = False
        scene.yaxis.visible = False
        scene.zaxis.visible = False
        # if(scale is not False):
        #     scene.xaxis.range = [-scale[0],scale[0]]
        #     scene.yaxis.range = [-scale,scale]
        #     scene.zaxis.range = [-scale,scale]

    def plotly_normals(self, points, normals):
        return go.Cone(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],  # i, j and k give the vertices of triangles
            u=normals[:, 0],
            v=normals[:, 1],
            w=normals[:, 2],
            # showlegend=False,
            showscale=False,
            hoverinfo="text",
            text=[str(idx) for idx in range(6890)],
            sizemode="scaled",
            sizeref=2,
            anchor="tip",
        )

    def plotly_mesh(self, source_mesh, color_map_vis, mesh_or_pc="mesh"):
        if mesh_or_pc == "mesh":
            return go.Mesh3d(
                x=source_mesh.vert[:, 0],
                y=source_mesh.vert[:, 1],
                z=source_mesh.vert[:, 2],  # i, j and k give the vertices of triangles
                i=source_mesh.face[:, 0],
                j=source_mesh.face[:, 1],
                k=source_mesh.face[:, 2],
                vertexcolor=color_map_vis,
                # showlegend=False,
                hoverinfo="text",
                text=[str(idx) for idx in range(6890)],
                lighting=dict(ambient=0.4, diffuse=0.6, roughness=0.9, ),
                lightposition=dict(z=5000),
            )
        else:
            try:
                color = ['rgb(' + str(int(c[0] * 255)) + ',' + str(int(255 * c[1])) + ',' + str(int(255 * c[2])) + ')'
                         for c in color_map_vis]
            except:
                color = ['rgb(' + str(0) + ',' + str(0) + ',' + str(0) + ')' for c in color_map_vis]

            return go.Scatter3d(
                x=source_mesh.vert[:, 0],
                y=source_mesh.vert[:, 1],
                z=source_mesh.vert[:, 2],  # i, j and k give the vertices of triangles
                mode='markers',
                marker=dict(
                    size=2,
                    color=color,  # set color to an array/list of desired values
                ),
                # showlegend=False,
                hoverinfo="text",
                text=[str(idx) for idx in range(6890)],
            )

    def visualize_scalar_vector_on_shape(
            self,
            source_mesh,
            scalar_represent_color_vector,
            max_scalar=None,
            mesh_or_pc='mesh',
            normals=None
    ):
        if not type(scalar_represent_color_vector).__module__ == np.__name__:
            
            scalar_represent_color_vector = (
                scalar_represent_color_vector.cpu().detach().numpy()
            )
        if normals is not None and not type(normals).__module__ == np.__name__:
            normals = normals.cpu().detach().numpy()

        scalar_represent_color_vector = scalar_represent_color_vector / (
                max_scalar or scalar_represent_color_vector.max()
        )
        Viridis = plt.get_cmap("viridis")
        colors = [Viridis(scalar) for scalar in scalar_represent_color_vector]
        if (normals is None):
            fig = go.Figure(
                data=[self.plotly_mesh(source_mesh=source_mesh, color_map_vis=colors, mesh_or_pc=mesh_or_pc), ]
            )
        else:
            fig = go.Figure(data=[self.plotly_normals(points=source_mesh.vert, normals=normals)])
        camera = dict(up=dict(x=0, y=1.0, z=0), eye=dict(x=-0.0, y=-0.0, z=5))

        with fig.batch_update():
            self.set_fig_settings(scene=fig.layout.scene, fig=fig, camera=camera)

        fig.update_layout(
            autosize=False, width=400, height=600,
        )

        if self.display_eng is not None:
            self.display_eng.stop()

        return fig

