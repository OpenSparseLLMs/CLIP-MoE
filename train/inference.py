import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from clipmoe import clipmoe
from dataset import DCDataset, share4v_train_dataset
import subprocess

def setup(backend="nccl", port=None):
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        os.environ["MASTER_PORT"] = "29522"
        os.environ["MASTER_ADDR"] = addr
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank % num_gpus)
    return rank


def main(rank,args):
    stage = args.stage
    expname = args.exp_name
    checkpoint_dir=args.checkpoint_dir
    savepath=args.save_path
    #num_clusters = args.num_clusters
    model, _ = clipmoe.load(checkpoint_dir)
    model=model.cuda()
    model = DDP(model, device_ids=[rank])
    
    #inference to obtain representations of all the training dataset
    if 'share' in args.dataset:
        my_dataset =share4v_train_dataset()
        print('load sharegpt4v dataset, total samples:')
        print(len(my_dataset))
    elif 'datacomp' in args.dataset.lower():
        my_dataset = DCDataset()
        print('load datacomp dataset, total samples:')
        print(len(my_dataset))
    my_dataset.output_idx=True
    sampler = torch.utils.data.distributed.DistributedSampler(my_dataset, shuffle=False)
    my_dataloader = DataLoader(my_dataset, batch_size=args.inference_batch_size, sampler=sampler, num_workers=args.num_workers,drop_last=False)
    image_features = []
    text_features=[]
    indices=[]
    model.eval()
    with torch.no_grad(),torch.cuda.amp.autocast():
        for img, texts,idx in tqdm(my_dataloader,disable=(rank != 0)):
            img = img.cuda()
            texts = clipmoe.tokenize(texts, truncate=True).cuda()
            image_feature = model.module.encode_image(img)
            text_feature=model.module.encode_text(texts)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            image_features.append(image_feature.cpu())
            text_feature/=text_feature.norm(dim=-1, keepdim=True)
            text_features.append(text_feature.cpu())

            indices.append(torch.LongTensor(idx))
    image_features = torch.cat(image_features,dim=0)
    text_features=torch.cat(text_features,dim=0)
    indices = torch.cat(indices,dim=0)
    os.makedirs(savepath,exist_ok=True)
    torch.save({'image_features':image_features,'text_features':text_features,'indices':indices},os.path.join(savepath, '{}_{}_{}_{}.pt'.format(expname, stage, 'features',rank)))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='encode all samples in training dataset')
    parser.add_argument('--stage', type=int, default=0, help='Stage of MCL, 0 for init vanilla model')
    parser.add_argument('--exp-name', type=str, default='clipmoe', help='Model architecture of open_clip')
    parser.add_argument('--dataset', type=str, default='datacomp', help='Dataset name')
    parser.add_argument('--checkpoint-dir', type=str, default='ckpt.pt', help='Checkpoint directory of the last stage model')
    parser.add_argument('--save-path', type=str, default='./save_mcl_tmp', help='Path to save the outputs')
    parser.add_argument('--inference-batch-size', type=int, default=1024, help='batch_size of the inference')
    parser.add_argument('--num-workers', type=int, default=8, help='num_workers of the inference')
    args = parser.parse_args()
    rank=setup()
    print("DDP setup Done")

    main(rank,args)
