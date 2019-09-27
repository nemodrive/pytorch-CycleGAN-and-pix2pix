import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from roadpackage.road import overlay_people_on_road
import pandas as pd
import torch
import cv2
from resizeimage import resizeimage
import random
import numpy as np


PEOPLE_PATH = "datasets/nemodrive/people_path.txt"
ROAD_PATH = "datasets/nemodrive/road_path.txt"
PATCH_PATH = "datasets/people_patch_path.txt"
PATCH_WIDTH = 256
PATCH_HEIGHT = 256
ROAD_COLOR = (0, 0, 0)


class NemodriveDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # CSV reader
        self.df_people = pd.read_csv(PEOPLE_PATH, header=None)
        self.df_road = pd.read_csv(ROAD_PATH, sep='\t', header=None)
        self.df_patch = pd.read_csv(PATCH_PATH, header=None)
        self.crop = self.opt.crop_size

        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        # self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.input_nc = 6
        # self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.output_nc = 3

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        # AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')

        idp = random.randrange(len(self.df_people) - 1)

        # A = human
        A = Image.open(self.df_people[0][idp]).convert('RGBA')
        A_path = self.df_people[0][idp]

        # set the patch height and width
        patch_height, patch_width = self.crop, self.crop

        # A = resizeA(A);

        # crop image A
        A = A.crop((A.size[0] - patch_height, A.size[1] - patch_width, A.size[0], A.size[1]))

        # TODO resize A
        # human always in image - max of human image to fit between 0.7 and 0.9
        # add one channel for mask

        # B = road , B_seg = segmented roads
        B = Image.open(self.df_road[0][index]).convert('RGB')
        B_without_human = B
        B_seg = Image.open(self.df_road[1][index]).convert('RGB')
        B_path = self.df_road[0][index]

        # get the coordinates of top-left B and crop
        x, y = self.road_coords(B_seg, patch_width, patch_height)
        B = B.crop((x - patch_height, y - patch_width, x, y))

        #get patch with people on road
        # idp = random.randrange(len(self.df_patch) - 1)
        idp = random.randrange(len(self.df_road) - 1)

        # C = Image.open(self.df_patch[0][idp]).convert('RGB')
        C = Image.open(self.df_road[0][idp]).convert('RGB')
        C_seg = Image.open(self.df_road[1][idp]).convert('RGB')

        x, y = self.road_coords(C_seg, patch_width, patch_height)
        C = C.crop((x - patch_height, y - patch_width, x, y))
        C_path = self.df_people[0][idp]

        A.load()
        B.paste(A, mask=A.split()[3])

        # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_without_human_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        C_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        # A = A_transform(A)
        B = B_transform(B)
        B_without_human = B_without_human_transform(B_without_human)
        C = C_transform(C)

        # print(type(B), B.size(), B_without_human.size())

        # B_pix = B.load()
        # B_human_pix = B_without_human.load()

        mask_human = (B_without_human == B).sum(dim = 0) == 3

        mask_human = mask_human.unsqueeze(0).expand(3, patch_height, patch_width)

        mask_human = 1 - mask_human

        mask_human = mask_human.type(torch.float)

        # print(mask_human.size())
        # for row in range(patch_height):
        #     for column in range(patch_width):
        #         if B_without_human[0, row, column] == B[0, row, column] and B_without_human[1, row, column] == B[1, row, column] and B_without_human[2, row, column] == B[2, row, column]:
        #             mask_human[0, row, column] = 1
        #             mask_human[1, row, column] = 1
        #             mask_human[2, row, column] = 1

        # return A
        # return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        # patch_with_road, cropped_obj, patch_with_obj_on_bg = torch.rand()
        return {'patch_with_bg': B,
                'patch_without_human': B_without_human,
                'mask': mask_human,
                # 'cropped_obj': A,
                'patch_with_obj_on_bg': C,
                'A_paths': A_path,
                'B_paths': B_path,
                'C_paths': C_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        # return len(self.AB_paths)
        return len(self.df_road)

    def road_coords(self, img: Image, patch_width: int, patch_height:int):
        road_pix_row = []
        road_pix_col = []
        image = img.load()
        for row in range(patch_height, img.size[0]):
            for column in range(patch_width, img.size[1]):
                if image[row, column] == ROAD_COLOR:
                    road_pix_row.append(row)
                    road_pix_col.append(column)

        id = random.randrange(len(road_pix_row) - 1)
        return road_pix_row[id], road_pix_col[id]
