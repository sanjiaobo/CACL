import os
import copy
import random
import numpy as np
from PIL import Image, ImageDraw,ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.utils.data as data

def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset_EM(data.Dataset):
    def __init__(self, imgs, labels, landmarks, bboxs, flag, needAU, Model, transform=None, target_transform=None, loader=default_loader):
        self.imgs = imgs
        self.labels = labels
        #self.landmarks = landmarks
        self.bboxs = bboxs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.flag = flag
        self.needAU = needAU
        self.Model = Model
        
    def __getitem__(self, index):
        img, label, bbox = self.loader(self.imgs[index]), copy.deepcopy(self.labels[index]), copy.deepcopy(self.bboxs[index])
        ori_img_w, ori_img_h = img.size

        # BoundingBox
        left  = bbox[0]
        upper = bbox[1]
        right = bbox[2]
        lower = bbox[3]

        # Visualization
        # draw = ImageDraw.Draw(img)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img.save('./vis_/{}.jpg'.format(index))

        enlarge_bbox = True

        if self.flag=='train':
            random_crop = True
            random_flip = True
        elif self.flag=='test':
            random_crop = False
            random_flip = False

        padding = max(0, int((right - left)*0.15)) # enlarge bbox
        half_padding = int(padding*0.5)
	
        if enlarge_bbox:
            left  = max(left - half_padding, 0)
            right = min(right + half_padding, ori_img_w)
            upper = max(upper - half_padding, 0)
            lower = min(lower + half_padding, ori_img_h)

        if random_crop:
            offset_range = half_padding

            x_offset = random.randint(-offset_range, offset_range)
            y_offset = random.randint(-offset_range, offset_range)

            left  = max(left + x_offset, 0)
            right = min(right + x_offset, ori_img_w)
            upper = max(upper + y_offset, 0)
            lower = min(lower + y_offset, ori_img_h)

        img = img.crop((left,upper,right,lower))
        crop_img_w, crop_img_h = img.size

        #landmark[:,0]-=left
        #landmark[:,1]-=upper

        # Visualization
        # draw = ImageDraw.Draw(img)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img.save('./vis_/{}.jpg'.format(index))

        if random_flip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            #landmark[:,0] = (right - left) - landmark[:,0]

        # Transform Image
        trans_img = self.transform(img)
        _, trans_img_w, trans_img_h = trans_img.size()
        
        if self.target_transform is not None:
            label = self.transform(label)

        # Don't need AU
        if not self.needAU:
            return (trans_img, self.imgs[index]), 0, label

        # get au location
       # landmark[:, 0] = landmark[:, 0] * trans_img_w / crop_img_w
       # landmark[:, 1] = landmark[:, 1] * trans_img_h / crop_img_h

        au_location = {}#get_au_loc(copy.deepcopy(landmark),trans_img_w, 56)
        # for i in range(au_location.shape[0]):
        #     for j in range(4):
        #         if au_location[i,j]<=11:
        #             au_location[i,j] = 12 
        #         if au_location[i,j]>=45: 
        #             au_location[i,j] = 44
        
        # Visualization
        # img_transform = img.resize((trans_img_w ,trans_img_h)) 
        # draw = ImageDraw.Draw(img_transform)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img_transform.save('./vis/{}.jpg'.format(index))
        
        #au_location = torch.LongTensor(au_location)

        return (trans_img, self.imgs[index]), au_location, label

    def __len__(self): 
        return len(self.imgs)

class MyDataset_EM_nobox(data.Dataset):
    def __init__(self, imgs, labels, landmarks, flag, needAU, Model, transform=None, target_transform=None, loader=default_loader):
        self.imgs = imgs
        self.labels = labels
        #self.landmarks = landmarks
        #self.bboxs = bboxs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.flag = flag
        self.needAU = needAU
        self.Model = Model
        
    def __getitem__(self, index):
        img = self.loader(self.imgs[index])
        label = copy.deepcopy(self.labels[index])
        ori_img_w, ori_img_h = img.size
        # BoundingBox
        left  = 0
        upper = 0
        right = ori_img_w
        lower = ori_img_h

        # Visualization
        # draw = ImageDraw.Draw(img)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img.save('./vis_/{}.jpg'.format(index))

        enlarge_bbox = True

        if self.flag=='train':
            random_crop = True
            random_flip = True
        elif self.flag=='test':
            random_crop = False
            random_flip = False

        padding = max(0, int((right - left)*0.15)) # enlarge bbox
        half_padding = int(padding*0.5)
	
        if enlarge_bbox:
            left  = max(left - half_padding, 0)
            right = min(right + half_padding, ori_img_w)
            upper = max(upper - half_padding, 0)
            lower = min(lower + half_padding, ori_img_h)

        if random_crop:
            offset_range = half_padding

            x_offset = random.randint(-offset_range, offset_range)
            y_offset = random.randint(-offset_range, offset_range)

            left  = max(left + x_offset, 0)
            right = min(right + x_offset, ori_img_w)
            upper = max(upper + y_offset, 0)
            lower = min(lower + y_offset, ori_img_h)

        img = img.crop((left,upper,right,lower))
        crop_img_w, crop_img_h = img.size

        #landmark[:,0]-=left
        #landmark[:,1]-=upper

        # Visualization
        # draw = ImageDraw.Draw(img)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img.save('./vis_/{}.jpg'.format(index))

        if random_flip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            #landmark[:,0] = (right - left) - landmark[:,0]

        # Transform Image
        trans_img = self.transform(img)
        _, trans_img_w, trans_img_h = trans_img.size()
        
        if self.target_transform is not None:
            label = self.transform(label)
        # Don't need AU
        if not self.needAU:
            return (trans_img, self.imgs[index]), 0, label

        # get au location
       # landmark[:, 0] = landmark[:, 0] * trans_img_w / crop_img_w
       # landmark[:, 1] = landmark[:, 1] * trans_img_h / crop_img_h

        au_location = {}#get_au_loc(copy.deepcopy(landmark),trans_img_w, 56)
        # for i in range(au_location.shape[0]):
        #     for j in range(4):
        #         if au_location[i,j]<=11:
        #             au_location[i,j] = 12 
        #         if au_location[i,j]>=45: 
        #             au_location[i,j] = 44
        
        # Visualization
        # img_transform = img.resize((trans_img_w ,trans_img_h)) 
        # draw = ImageDraw.Draw(img_transform)
        # for idx in range(len(landmark[:,0])):
        #     draw.point((landmark[idx, 0], landmark[idx, 1]))
        # img_transform.save('./vis/{}.jpg'.format(index))
        
        #au_location = torch.LongTensor(au_location)

        return (trans_img, self.imgs[index]), au_location, label

    def __len__(self): 
        return len(self.imgs)


