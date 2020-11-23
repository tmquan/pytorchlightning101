import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import kornia
list_pathologies = ["Covid"]
list_pathologies = sorted(list_pathologies)
class CustomDataset(Dataset):
    """
    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.
    Jeremy Irvin *, Pranav Rajpurkar *, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, 
    Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong, 
    Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson, Curtis P. Langlotz, 
    Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng. https://arxiv.org/abs/1901.07031
    
    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/
    """
    def __init__(self, imgpath, csvpath, 
                 pathology='Covid', 
                 transform=None, 
                 is_train='train',
                 size=500000,
                 seed=2020):

        super(CustomDataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.MAXVAL = 255
        self.size=500000
        self.pathology = pathology
        self.is_train = is_train=='train'
        self.imgpath = imgpath
        self.transform = transform
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)

        # Get our classes.
        self.positives = self.csv.copy(deep=True)[self.csv[self.pathology] >=0.5].reset_index() 
        self.negatives = self.csv.copy(deep=True)[self.csv[self.pathology] < 0.5].reset_index()
        
        print(len(self.positives))
        print(len(self.negatives))
        
    def __repr__(self):
        pass
    
    def __len__(self):
        return len(self.csv) if len(self.csv) < self.size else self.size

    def __getitem__(self, idx):
        if self.is_train:
            pos_idx = np.random.randint(len(self.positives))
            pos_id = self.positives['Images'].iloc[pos_idx]
            pos_image_path = os.path.join(self.imgpath, pos_id)
            pos_image = cv2.imread(pos_image_path, cv2.IMREAD_GRAYSCALE)

            neg_idx = np.random.randint(len(self.negatives))        
            neg_id = self.negatives['Images'].iloc[neg_idx]
            neg_image_path = os.path.join(self.imgpath, neg_id)
            neg_image = cv2.imread(neg_image_path, cv2.IMREAD_GRAYSCALE)

            if pos_image is None:
                print(pos_image_path)
            if neg_image is None:
                print(neg_image_path)
            def bbox2(img):
                #print(img.shape)
                rows = np.any(img, axis=1)
                cols = np.any(img, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                return img[ymin:ymax+1, xmin:xmax+1]
                
            pos_image = bbox2(pos_image)
            neg_image = bbox2(neg_image) # Crop to valid part
            if self.transform is not None:
                pos_transformed = self.transform(image=pos_image)
                pos_image = pos_transformed['image']
                neg_transformed = self.transform(image=neg_image)
                neg_image = neg_transformed['image']


            return kornia.image_to_tensor(pos_image).float(), \
                   torch.Tensor([self.positives[self.pathology].iloc[pos_idx]]).float(), \
                   kornia.image_to_tensor(neg_image).float(), \
                   torch.Tensor([self.negatives[self.pathology].iloc[neg_idx]]).float()
        else:
            ord_idx = idx #np.random.randint(len(self.csv))        
            ord_id = self.csv['Images'].iloc[ord_idx]
            ord_image_path = os.path.join(self.imgpath, ord_id)
            ord_image = cv2.imread(ord_image_path, cv2.IMREAD_GRAYSCALE)
            # print(ord_image_path)
            if self.transform is not None:
                ord_transformed = self.transform(image=ord_image)
                ord_image = ord_transformed['image']
                
            return kornia.image_to_tensor(ord_image).float(), \
                   torch.Tensor([self.csv[self.pathology].iloc[ord_idx]]).float(), \
                   kornia.image_to_tensor(ord_image).float(), \
                   torch.Tensor([self.csv[self.pathology].iloc[ord_idx]]).float()



if __name__ == '__main__':
    ds = CustomDataset(imgpath='/raid/data/COVID_Data_Relabel/data/',
                       csvpath='train_covid_quan.csv',
                       is_train='valid'
                       )
    for i in range(10):
        sample = ds[i]

        print(i, 
              sample[0].shape, 
              sample[1], 
              # sample[2].shape, 
              # sample[3], 
              )

    # ds = CustomDataset(imgpath='/raid/data/COVID_Data_Relabel/data/',
    #                    csvpath='valid_covid_quan.csv',
    #                    is_train='test'
    #                    )
    # for i in range(10):
    #     sample = ds[i]

    #     print(i, 
    #           sample[0].shape, 
    #           sample[1]
    #           )