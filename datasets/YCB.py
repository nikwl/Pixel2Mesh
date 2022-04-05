import json
import os
import pickle

import numpy as np
import torch
from PIL import Image, ImageOps
from skimage import io, transform
from torch.utils.data.dataloader import default_collate
import trimesh
import scipy.io as scio

import config
from datasets.base_dataset import BaseDataset


def normalize_unit_cube(mesh, scale=True):
    """Normalize a mesh so that it occupies a unit cube"""

    # Get the overall size of the object
    mesh = mesh.copy()
    mesh_min, mesh_max = np.min(mesh.vertices, axis=0), np.max(mesh.vertices, axis=0)
    size = mesh_max - mesh_min

    # Center the object
    mesh.vertices = mesh.vertices - ((size / 2.0) + mesh_min)

    # Normalize scale of the object
    if scale:
        mesh.vertices = mesh.vertices * (1.0 / np.max(size))
    try:
        mesh.fix_normals()
    except AttributeError:
        pass
    return mesh


class YCB(BaseDataset):
    """
    """

    def __init__(self, file_root, file_list_name, mesh_pos, normalization, ycb_options):
        super().__init__()
        
        self.root = file_root
        self._do_caching = True
        self._cache = {}

        # Load models
        self._model_dir = os.path.join(self.root, "models")
        class_file = open(os.path.join(self.root, "dataset_config", "classes.txt"))
        class_file = open(os.path.join(self.root, "dataset_config", "classes_subset.txt.old"))
        self._model_list = {}
        print("Loading models...")
        self.targets = [3, 10, 13, 15, 16]
        for class_id in self.targets:
            class_input = class_file.readline()
            if not class_input:
                break
            
            # Load that model
            model = self.load(
                os.path.join(self._model_dir, class_input[:-1], "textured.obj")
            )
            # Make sure normals are correct
            model = normalize_unit_cube(model)
            model.fix_normals()

            # verts, norm_inds = model.sample(1000, return_index=True)
            # norms = model.face_normals[norm_inds, :]
            verts = np.array(model.vertices)
            norms = np.array(model.vertex_normals)

            self._model_list[class_id] = [verts, norms]
    
        print("Loaded {} models".format(len(self._model_list)))

        # Load train/test file list
        if "test" in file_list_name:
            file_list_name = "test_data_list"
        elif "train" in file_list_name:
            file_list_name = "train_data_list_subset_half"
        else:
            raise RuntimeError()
        input_file = open(os.path.join(self.root, "dataset_config", file_list_name + ".txt"), "r")
        print("Reading file list...")
        self.file_names = []
        while True:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]

            # Full image path
            self.file_names.append(
                input_line
            )
        input_file.close()

        inds = np.random.randint(0, len(self.file_names), size=(10000,))
        self.file_names = [self.file_names[i] for i in inds]

        self._do_caching = False

        self.normalization = normalization
        self.mesh_pos = mesh_pos
        self.resize_with_constant_border = ycb_options.resize_with_constant_border

        self.load_npy()

    def load(self, f_in):
        """
        An autocaching loader
        """
        if f_in not in self._cache:
            ext = os.path.splitext(f_in)[-1]
            if ext == ".png":
                data = Image.open(f_in)
            elif ext == ".xyz":
                data = np.loadtxt(f_in, delimiter=" ")
            elif ext == ".ply":
                data = trimesh.load(f_in)
            elif ext == ".mat":
                data = scio.loadmat(f_in)
            elif ext == ".obj":
                data = trimesh.load(f_in)
            else:
                raise RuntimeError("Unknown extension: {}".format(ext))
            if not self._do_caching:
                return data
            self._cache[f_in] = data
        
        return self._cache[f_in]

    def load_npy(self):
        print("Loading numpy data ...")
        self.list_rgb = np.load(self.root + "/list_rgb.npy")
        # assert len(self.list) == len(self.list_rgb)
        # self.list_depth = np.load(self.root + "/list_depth.npy")
        self.list_label = np.load(self.root + "/list_label.npy")
        self.list_meta = np.load(self.root + "/list_meta.npy", allow_pickle=True)

    def __getitem__(self, index):
        # print("Getting ", index)
        # 0 = 4.45
        # 1 = 3.8
        # 2 = 

        if hasattr(self, "list_rgb"):
            img = self.list_rgb[:, :, :, index]
            # depth = self.list_depth[:, :, index]
            label = self.list_label[:, :, index]
            meta = self.list_meta[index]
        else:
            try:
                img = np.array(self.load('{0}/{1}-color.png'.format(self.root, self.file_names[index])))
                # depth = np.array(self.load('{0}/{1}-depth.png'.format(self.root, self.file_names[index])))
                label = np.array(self.load('{0}/{1}-label.png'.format(self.root, self.file_names[index])))
                meta = self.load('{0}/{1}-meta.mat'.format(self.root, self.file_names[index]))
            except FileNotFoundError:
                print("FileNotFoundError: {}/{}".format(self.root, self.file_names[index]))
                return self[index+1]

        # Get the indices of all models in the image
        obj = meta['cls_indexes'].flatten().astype(np.int32)

        # Pick a random model index
        greenlit = list(set(self.targets).intersection(set(obj)))
        idx = greenlit[np.random.randint(0, len(greenlit))]

        # Get the points and normals for that model
        model = self._model_list[idx]
        pts = np.array(model[0]).astype(np.float32)
        normals = np.array(model[1]).astype(np.float32)

        # Get the masked image
        mask = np.ma.getmaskarray(np.ma.masked_equal(label, idx))
        img = img[:, :, :3]
        img = img * np.expand_dims(mask, axis=2) # Remove pixels in masked regions
        img = img.astype(np.float32) / 255.0 # Convert to {0, 1}
        img = img + np.logical_not(np.expand_dims(mask, axis=2)).astype(float) # Convert maksed regions to white
        
        # Resize with padding
        img = (img * 255).astype(np.uint8)
        aspect_ratio = img.shape[0] / img.shape[1]
        img = np.array(ImageOps.pad(Image.fromarray(img).resize((227, int(227 * aspect_ratio))), (227, 227), color="white"))
        img = img.astype(np.float32) / 255.0 # Convert to {0, 1}

        pts -= np.array(self.mesh_pos)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized,
            "images_orig": img,
            "points": pts,
            "normals": normals,
            "labels": idx,
            "filename": self.file_names[index],
            "length": length
        }

    def __len__(self):
        return len(self.file_names)


def get_ycb_collate(num_points):
    """
    :param num_points: This option will not be activated when batch size = 1
    :return: ycb_collate function
    """
    def ycb_collate(batch):
        if len(batch) > 1:
            all_equal = True
            for t in batch:
                if t["length"] != batch[0]["length"]:
                    all_equal = False
                    break
            points_orig, normals_orig = [], []
            if not all_equal:
                for t in batch:
                    pts, normal = t["points"], t["normals"]
                    length = pts.shape[0]
                    choices = np.resize(np.random.permutation(length), num_points)
                    t["points"], t["normals"] = pts[choices], normal[choices]
                    points_orig.append(torch.from_numpy(pts))
                    normals_orig.append(torch.from_numpy(normal))
                ret = default_collate(batch)
                ret["points_orig"] = points_orig
                ret["normals_orig"] = normals_orig
                return ret
        ret = default_collate(batch)
        ret["points_orig"] = ret["points"]
        ret["normals_orig"] = ret["normals"]
        return ret

    return ycb_collate