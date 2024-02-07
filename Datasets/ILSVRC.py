import os
from glob import glob
import PIL
import torch
from torchvision.datasets import ImageFolder
from bs4 import BeautifulSoup

class ImageNetDataset_val(ImageFolder):
    def __init__(self, root_dir, transforms=None):
        self.img_dir = os.path.join(root_dir, "Data", "CLS-LOC", "val")
        self.annotation_dir = os.path.join(root_dir, "Annotations", "CLS-LOC", "val")
        self.classes = os.listdir(self.img_dir)
        self.transforms = transforms
        self.img_data = []
        self.img_labels = []

        for idx, cls in enumerate(self.classes):
            # self.class_name.append(cls)
            img_cls_dir = os.path.join(self.img_dir, cls)

            for img in glob(os.path.join(img_cls_dir, '*.jpeg')):
                self.img_data.append(img)
                self.img_labels.append(idx)


    def __getitem__(self, idx):
        img_path, label = self.img_data[idx], self.img_labels[idx]
        img = PIL.Image.open(img_path).convert('RGB')
        # img.show()
        width, height = img.size
        img_name = img_path.split('\\')[-1].split('.')[0]
        anno_path = os.path.join(self.annotation_dir, img_name+".xml")
        with open(anno_path, 'r') as f:
            file = f.read()
        soup = BeautifulSoup(file, 'html.parser')
        if self.transforms:
            img = self.transforms(img)
        objects = soup.findAll('object')
        
        bnd_box = torch.tensor([])

        for object in objects:
            xmin = int(object.bndbox.xmin.text)
            ymin = int(object.bndbox.ymin.text)
            xmax = int(object.bndbox.xmax.text)
            ymax = int(object.bndbox.ymax.text)
            xmin = int(xmin/width*224)
            ymin = int(ymin/height*224)
            xmax = int(xmax/width*224)
            ymax = int(ymax/height*224)
            if bnd_box.dim()==1:
                bnd_box = torch.tensor((xmin, ymin, xmax, ymax)).unsqueeze(0)
            else:
                bnd_box = torch.cat((bnd_box, torch.tensor((xmin, ymin, xmax, ymax)).unsqueeze(0)), dim=0)
        # print(bnd_box.shape)
        sample = {'image': img, 'label': label, 'filename': img_name, 'num_objects': len(objects), 'bnd_box': bnd_box, 'img_path': img_path}
        return sample

    def __len__(self):
        return len(self.img_data)
