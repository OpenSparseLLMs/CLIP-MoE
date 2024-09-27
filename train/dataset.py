import json
from PIL import Image
import clip
import torch
import torch.utils.data as data
import os
import torch.distributed as dist


data4v_root = 'sharegpt4v/data/'
json_name = 'share-captioner_coco_lcs_sam_1246k_1107.json'
image_root = 'sharegpt4v/data/'
resolution='ViT-L/14' #define the preprocess as used in CLIP. for 336px use "ViT-L/14@336px"

class share4v_val_dataset(data.Dataset):
    def __init__(self):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 10000
        with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)[:self.total_len]
        _ , self.preprocess = clip.load(resolution)
    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        caption_short = caption.split(". ")[0]
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, caption, caption_short

class DCDataset(data.Dataset):
    def __init__(
            self,
            data_root='', #absolute path to the root of the dataset
            transform=None,
            tokenizer=None,
    ):
        folder_names = [name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name))]
        folder_names=sorted(folder_names)
        self.img_file_list = []
        self.caption_file_list = []
        for directory in folder_names:
            directory = os.path.join(data_root, directory)
            self.img_file_list += sorted([os.path.join(directory,filename) for filename in os.listdir(directory) if filename.endswith('.jpg')])
            self.caption_file_list += sorted([os.path.join(directory,filename) for filename in os.listdir(directory) if filename.endswith('.txt')])
        self.img_file_list = self.img_file_list[:1000000]
        self.caption_file_list = self.caption_file_list[:1000000]
        _ , self.preprocess = clip.load(resolution)
        self.transform=transform
        self.tokenizer=tokenizer
        self.output_idx=False
        
    def __len__(self):
        return len(self.img_file_list)
    
    def __getitem__(self, idx):
        img_file=self.img_file_list[idx]
        caption_file=self.caption_file_list[idx]
        img = Image.open(img_file).convert('RGB')
        img=self.preprocess(img)
        with open(caption_file, 'r') as f:
            caption = f.read()
        if self.transform is not None:
            img = self.transform(img)
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)[0]
        if self.output_idx:
            return img, caption, idx
        return img, caption,caption


    
class share4v_train_dataset(data.Dataset):
    def __init__(self):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 10000
        with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)[self.total_len:]
        _ , self.preprocess = clip.load(resolution)
        self.output_idx=False

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")

        caption_short = caption.split(". ")[0]
        
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        if self.output_idx:
            return image_tensor, caption, index
        return image_tensor, caption, caption_short

class MCL_batch_sampler_DDP():
        
    def __init__(self, pseudo_label, batch_size,cluster_num):
        self.batch_size = batch_size
        self.cluster_num=cluster_num
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.pseudo_label=pseudo_label
     
        self.original_modulo_clusters = {i: [] for i in range(self.cluster_num)} 
        # Group the dataset indices by their pseudo-label values
        for idx, cluster in enumerate(pseudo_label):
            self.original_modulo_clusters[cluster.item()].append(idx)  

    def __iter__(self):
        
        # Evenly distribute the data according to pseudo-labels to each GPU process
       
        self.modulo_clusters = {i: [] for i in range(self.cluster_num)} 
        
        for i in range(self.cluster_num):
            # Ensure randomness in each epoch
            rand_idx=torch.randperm(len(self.original_modulo_clusters[i]))
            for j in range(len(rand_idx)//self.world_size//self.batch_size*self.batch_size*self.world_size):
                if j % self.world_size == self.rank:
                    self.modulo_clusters[i].append(self.original_modulo_clusters[i][rand_idx[j]])
      
        batch = []
        modulo = 0
        while any(len(cluster) > 0 for cluster in self.modulo_clusters.values()): # Check if there are any indices left
        # The loop works as follows: suppose there are five clusters, then the 0th batch samples from the cluster with label=0; the 1st batch samples from the cluster with label=1; and so on. 
        # If a particular cluster does not have enough samples for one batch, it is skipped.
        # The current loop scheme is naive. Better schedule can be arranged.
            if len(self.modulo_clusters[modulo]) > 0:
                batch.append(self.modulo_clusters[modulo].pop())
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    modulo = (modulo + 1) % self.cluster_num
            else:
                batch = []
                modulo = (modulo + 1) % self.cluster_num

    def __len__(self):
        return sum(len(cluster)//self.world_size//self.batch_size for cluster in self.original_modulo_clusters.values())
    