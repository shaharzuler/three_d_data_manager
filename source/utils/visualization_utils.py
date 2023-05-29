#TODO credits
#todo add override attribute to dataset

from ast import Tuple
from dataclasses import dataclass
# from three_d_data_manager.source.file_paths import FilePaths
from .mesh_utils import MeshContainer

import torchvision
import os
from PIL import Image
import torch
import subprocess
import tqdm
from xvfbwrapper import Xvfb
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from PIL import Image
import cv2


def visualize_grid_of_lbo(verts ,faces ,lbos ,max_lbos=None ,dirpath='/home/oefroni/dfaust_project/plots/' ,
                          mesh_or_pc='mesh', prefix=''):
    imgs = []
    max_lbos = max_lbos or 14
    for eigvec_idx in tqdm(range(min(max_lbos, lbos.shape[1]))):
        MeshVisualizer(dataset="").visualize_scalar_vector_on_shape(
            MeshContainer(verts, faces),
            lbos[: ,eigvec_idx],
            save_path=os.path.join(dirpath, f'{prefix}_{eigvec_idx}.png'),
            mesh_or_pc=mesh_or_pc,
            write_html=True
            # max_scalar=lbos.max().item()
        )
        imgs.append(torchvision.transforms.functional.to_tensor(Image.open(os.path.join(dirpath,
                                                                                        f'{prefix}_{eigvec_idx}.png'))))

    torchvision.utils.save_image(torch.stack(imgs), nrow=2, fp=os.path.join(dirpath, prefix + '_lbo_grid.png'),)
    return os.path.join(dirpath,'lbo_grid.png')

def check_xvfb():

    comm = 'ps -ef | grep Xvfb | grep -v grep | cat'
    output = subprocess.check_output(comm, shell=True)

    if len(output) == 0:
        return False
    else:
        return True

def create_colormap(vert):
    """
    Creates a uniform color map on a mesh

    Args:
        VERT (Nx3 ndarray): The vertices of the object to plot

    Returns:
        Nx3: The RGB colors per point on the mesh
    """
    vert = np.double(vert)
    minx = np.min(vert[:, 0])
    miny = np.min(vert[:, 1])
    minz = np.min(vert[:, 2])
    maxx = np.max(vert[:, 0])
    maxy = np.max(vert[:, 1])
    maxz = np.max(vert[:, 2])
    colors = np.stack([((vert[:, 0] - minx) / (maxx - minx)), ((vert[:, 1] - miny) /
                                                               (maxy - miny)), ((vert[:, 2] - minz) / (maxz - minz))]).transpose()
    return colors


class MeshVisualizer:
    """Visualization class for meshs."""

    def __init__(self, dataset="FAUST", display_up=False):
        """
        The mesh visualization utility class

        Args:
            dataset (str, optional): Name of the dataset, e.g. will effect view angle . Defaults to 'FAUST'.
        """
        self.dataset = dataset
        self.scale = 0.8 if 'faust' in dataset else 0.2
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
            # lighting=dict(ambient=0.4, diffuse=0.6, roughness=0.9,),
            # lightposition=dict(z=5000),
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
            save_path="output/tmp/vis_scalar_on_shape.png",
            max_scalar=None,
            write_html=True,
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
        if write_html:
            fig.write_html(save_path + ".html")
        if save_path is not None:
            fig.write_image(save_path)

        if self.display_eng is not None:
            self.display_eng.stop()

        return fig


@dataclass
class colors:
    YELLOW_RGB:Tuple = 255, 255, 1
    RED_RGB:Tuple    = 255, 27,  62
    PURPLE_RGB:Tuple = 255, 1,   255
    BLUE_RGB:Tuple   = 0,   102, 255
    GREEN_RGB:Tuple  = 1,   255, 1


def draw_2d_sections(arr:np.array, save_path:str, pad_val = 10):
    arr = arr.astype(float)
    min_val, max_val = arr.min(), arr.max()
    arr_sections = np.array(arr.shape)//2
    x_section = arr[ arr_sections[0], :              , :               ]
    y_section = arr[ :              , arr_sections[1], :               ]
    z_section = arr[ :              , :              , arr_sections[2] ]
    sections_image_top = np.hstack((x_section, np.ones((arr.shape[0], arr.shape[1] + pad_val)) * min_val))
    sections_image_bottom = np.hstack((y_section, np.ones((arr.shape[0], pad_val)) * min_val, z_section))
    pad_top_from_bottom =  np.ones((pad_val, arr.shape[2] + pad_val + arr.shape[0])) * min_val
    sections_image = np.vstack((sections_image_top, pad_top_from_bottom, sections_image_bottom))
    sections_image = (((sections_image - min_val) / (max_val - min_val)) * 255).astype(np.uint8)

    if save_path is not None:
        Image.fromarray(sections_image).save(save_path, quality=100) 

    return sections_image

def draw_2d_mask_on_scan(sections_image:np.array, sections_mask:np.array, color:Tuple, save_path:str) -> np.array:
    if len(sections_image.shape) == 2:
        img_rgb = cv2.cvtColor(sections_image, cv2.COLOR_GRAY2RGB) # H x W x 3
    sections_mask = sections_mask.astype(float)/255
    mask_3_channels = np.repeat(np.expand_dims(sections_mask, -1), 3, axis=2) # H x W x 3
    mask_rgb = mask_3_channels.copy()
    mask_rgb[:,:,0] *= color[0]
    mask_rgb[:,:,1] *= color[1]
    mask_rgb[:,:,2] *= color[2]
    mask_rgb = mask_rgb.astype(sections_image.dtype)

    img_w_mask = np.where(mask_3_channels != 0, mask_rgb, img_rgb)
    if save_path is not None:
        Image.fromarray(img_w_mask, mode="RGB").save(save_path, quality=100) 

    return img_w_mask

def save_mask_and_image(image, mask, color, save_path):
    sections_mask = draw_2d_sections(mask, save_path=None)
    sections_image_w_mask = draw_2d_mask_on_scan(image, sections_mask, color, save_path) 
    return sections_image_w_mask

def draw_masks_and_contours(sections_image:np.array, masks_data:list[dict[str, any((str, np.array, tuple))]], target_root_dir:str, file_paths):#:FilePaths) -> FilePaths:
    contours_arr = cv2.cvtColor(sections_image, cv2.COLOR_GRAY2BGR)
    for n, mask_data in enumerate(masks_data):
        contours_arr, file_paths = draw_single_mask_and_contour(sections_image, contours_arr, n, mask_data, target_root_dir, file_paths)
    
    section_contours_path = os.path.join(target_root_dir, "section_contours.jpg")
    file_paths.section_contours = section_contours_path
    Image.fromarray(cv2.cvtColor(contours_arr, cv2.COLOR_BGR2RGB), mode="RGB").save(section_contours_path, quality=100) 
    
    return file_paths

def draw_single_mask_and_contour(sections_image:np.array, contours_arr:np.array, n:int, mask_data:list[dict[str, any((str, np.array, tuple))]], target_root_dir:str, file_paths):#:FilePaths)-> tuple[np.array, FilePaths]:
    mask_sections_path = os.path.join(target_root_dir, f"{n}_{mask_data['name']}.jpg")
    setattr(file_paths, f"{n}_{mask_data['name']}", mask_sections_path)

    sections_mask = draw_mask(sections_image, mask_data, mask_sections_path) 
    contours_arr = draw_mask_contours(contours_arr, n, mask_data, sections_mask)

    return contours_arr, file_paths

def draw_mask_contours(contours_arr, n, mask_data, sections_mask):
    contours, hierarchy = cv2.findContours(sections_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contours_arr, contours, -1, mask_data["color"], 1)
    cv2.putText(contours_arr, text=mask_data['name'], org=(150,(n+1)*10),fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=mask_data["color"], thickness=1, lineType=cv2.LINE_AA)
    return contours_arr

def draw_mask(sections_image, mask_data, mask_sections_path):
    sections_mask = draw_2d_sections(mask_data["arr"], save_path=None)
    sections_image_w_mask = draw_2d_mask_on_scan(sections_image, sections_mask, mask_data["color"], mask_sections_path)
    return sections_mask







#region meshvisualizer commented out
    # def visualize_pcs(
    #         self,
    #         pcs,
    #         save_path="output/tmp/P_mesh_pair.png",
    #         write_html=True,
    #         horiz_space=0,
    # ):
    #     if save_path is not None:
    #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     for idx in range(len(pcs)):
    #         if not type(pcs[idx]).__module__ == np.__name__:
    #             pcs[idx] = pcs[idx].detach().cpu().numpy()

    #     fig = make_subplots(
    #         rows=1,
    #         cols=len(pcs),
    #         shared_yaxes=True,
    #         shared_xaxes=True,
    #         specs=[[{"type": "scatter3d"} for i in range(len(pcs))]],
    #         horizontal_spacing=horiz_space,
    #     )
    #     for idx, pc in enumerate(pcs):
    #         fig.add_trace(
    #             self.plotly_mesh(
    #                 MeshContainer(vert=pc),
    #                 color_map_vis=create_colormap(pc),
    #                 mesh_or_pc="pc",
    #             ),
    #             row=1,
    #             col=idx + 1,
    #         )

    #     camera = dict(up=dict(x=0, y=1.0, z=0), eye=dict(x=-0.0, y=-0.0, z=5))

    #     with fig.batch_update():
    #         for idx in range(len(pcs)):
    #             # scale = np.max((np.max(np.abs(target_mesh.vert),0),np.max(np.abs(source_mesh.vert),0)),0)
    #             self.set_fig_settings(scene=eval(f"fig.layout.scene{idx + 1 if idx > 0 else ''}"), fig=fig,
    #                                   camera=camera, scale=self.scale)

    #     fig.update_layout(
    #         margin=dict(l=0, r=0, t=0, b=0),
    #         autosize=False, width=600 * len(pcs), height=900,
    #     )

    #     fig.write_image(save_path, )
    #     if write_html:
    #         fig.write_html(save_path + ".html")
    #     return fig

    # def visualize_pc_pair_same_fig(
    #         self,
    #         pcs,
    #         save_path="output/tmp/P_pc_pair_same_fig.png",
    #         write_html=True,
    #         horiz_space=0,
    # ):
    #     if save_path is not None:
    #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     for idx in range(len(pcs)):
    #         if not type(pcs[idx]).__module__ == np.__name__:
    #             pcs[idx] = pcs[idx].detach().cpu().numpy()

    #     fig = make_subplots(
    #         rows=1,
    #         cols=1,
    #         shared_yaxes=True,
    #         shared_xaxes=True,
    #         specs=[[{"type": "scatter3d"} for i in range(1)]],
    #         horizontal_spacing=horiz_space,
    #     )
    #     green, blue = (0, 0.45, 0), (0, 0, 1)
    #     for idx, pc in enumerate(pcs):
    #         color = green if idx == 0 else blue
    #         fig.add_trace(
    #             self.plotly_mesh(
    #                 MeshContainer(vert=pc),
    #                 color_map_vis=np.stack([color for i in range(pc.shape[0])]),
    #                 mesh_or_pc="pc",
    #             ),
    #             row=1,
    #             col=1,
    #         )

    #     camera = dict(up=dict(x=0, y=1.0, z=0), eye=dict(x=-0.0, y=-0.0, z=5))

    #     with fig.batch_update():
    #         for idx in range(1):
    #             # scale = np.max((np.max(np.abs(target_mesh.vert),0),np.max(np.abs(source_mesh.vert),0)),0)
    #             self.set_fig_settings(scene=eval(f"fig.layout.scene{idx + 1 if idx > 0 else ''}"), fig=fig,
    #                                   camera=camera, scale=self.scale)

    #     fig.update_layout(
    #         margin=dict(l=0, r=0, t=0, b=0),
    #         autosize=False, width=600, height=900,
    #     )

    #     fig.write_image(save_path, )
    #     if write_html:
    #         fig.write_html(save_path + ".html")
    #     return fig

    # def visualize_mesh_pair(
    #         self,
    #         source_mesh,
    #         target_mesh,
    #         corr_map,
    #         title=None,
    #         color_map=None,
    #         is_show=False,
    #         save_path="output/tmp/P_mesh_pair.png",
    #         same_fig=True,
    #         write_html=True,
    #         mesh_or_pc='mesh',
    #         horiz_space=0.1,
    #         grayed_indices=None,
    #         target_grayed=None,
    # ):
    #     if save_path is not None:
    #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     color_map_vis = color_map or create_colormap(source_mesh.vert)
    #     if (grayed_indices is not None):
    #         color_map_vis[grayed_indices] = 0
    #     fig = make_subplots(
    #         rows=1,
    #         cols=2,
    #         shared_yaxes=True,
    #         shared_xaxes=True,
    #         specs=[[{"type": "mesh3d"}, {"type": "mesh3d"}]],
    #         horizontal_spacing=horiz_space,
    #     )
    #     fig.add_trace(
    #         self.plotly_mesh(
    #             source_mesh,
    #             np.concatenate(
    #                 [color_map_vis, np.ones(color_map_vis.shape[0])[:, None]], axis=1
    #             ),
    #             mesh_or_pc=mesh_or_pc,
    #         ),
    #         row=1,
    #         col=1,
    #     )
    #     target_colors = color_map_vis[corr_map]
    #     if (target_grayed is not None):
    #         target_colors[target_grayed] = 0

    #     fig.add_trace(
    #         self.plotly_mesh(
    #             target_mesh,
    #             np.concatenate(
    #                 [target_colors, np.ones(corr_map.shape[0])[:, None]],
    #                 axis=1,
    #             ),
    #             mesh_or_pc=mesh_or_pc,
    #         ),
    #         row=1,
    #         col=2,
    #     )

    #     camera = dict(up=dict(x=0, y=1.0, z=0), eye=dict(x=-0.0, y=-0.0, z=5))

    #     with fig.batch_update():

    #         # scale = np.max((np.max(np.abs(target_mesh.vert),0),np.max(np.abs(source_mesh.vert),0)),0)
    #         self.set_fig_settings(scene=fig.layout.scene, fig=fig, camera=camera, scale=self.scale)
    #         self.set_fig_settings(scene=fig.layout.scene2, fig=fig, camera=camera, scale=self.scale)

    #     fig.update_layout(
    #         margin=dict(l=0, r=0, t=0, b=0),
    #         autosize=False, width=1200, height=900,
    #     )
    #     if save_path is not None:
    #         fig.write_image(save_path, )
    #     if write_html:
    #         fig.write_html(save_path + ".html")

    #     if self.display_eng is not None:
    #         self.display_eng.stop()

    #     return fig



    # def scale_by_dataset(self, mesh):
    #     """For specific datasets we scale the shape for better visualization."""
    #     if self.dataset == "TOSCA":
    #         mesh.vert = mesh.vert * 100
    #     return mesh


    # @staticmethod
    # def visualize_scalar_vector_on_shape_static(verts, face, scalar_map, display_up=False, extra_text="", normals=None,
    #                                             max_scalar=None, ):
    #     return MeshVisualizer(dataset='faust', display_up=display_up).visualize_scalar_vector_on_shape(
    #         source_mesh=MeshContainer(verts, face),
    #         scalar_represent_color_vector=scalar_map,
    #         save_path=f"output/tmp/{extra_text}_.png",
    #         mesh_or_pc='mesh' if face is not None else 'pc',
    #         normals=normals,
    #         max_scalar=None,
    #     )
#endregion meshvisualizer commented out