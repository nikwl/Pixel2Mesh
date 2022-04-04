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
        self._model_list = {}
        class_id = 1
        print("Loading models...")
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            
            # Load that model
            model = self.load(
                os.path.join(self._model_dir, class_input[:-1], "textured.obj")
            )
            # Make sure normals are correct
            model.fix_normals()
            verts, norm_inds = model.sample(1000, return_index=True)
            norms = model.face_normals[norm_inds, :]
            self._model_list[class_id] = [verts, norms]
    
            class_id += 1
        print("Loaded {} models".format(len(self._model_list)))

        # Load train/test file list
        assert file_list_name == "test" or file_list_name == "train"
        input_file = open(os.path.join(self.root, "dataset_config", file_list_name + "_data_list.txt"), "r")
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

    def __getitem__(self, index):
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
        idx = obj[np.random.randint(0, len(obj))]

        # Get the points and normals for that model
        idx = 1
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