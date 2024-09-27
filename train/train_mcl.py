from cgitb import text
import torch
from utils import concat_all_gather, is_dist_avail_and_initialized, accuracy
import torch.distributed as dist
from tqdm import tqdm
from dataset import share4v_train_dataset, MCL_batch_sampler_DDP,DCDataset
from clipmoe import clipmoe
from scheduler import cosine_lr
import argparse
import os
import subprocess
import torch.optim as optim
from torch.cuda.amp import GradScaler
# import warnings
# warnings.filterwarnings("ignore")


class CLIP_Clean_Train():
    def __init__(self, rank,local_rank,args):
        self.args=args
        self.rank=rank
        self.local_rank = local_rank
        self.MCL_label_path=args.MCL_label_path
        self.base_model = args.base_model
        self.model, _ = clipmoe.load_from_clip(self.base_model, device='cpu')
        self.model.train()
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)  
        self.model.use_short_text=args.use_short_text
        self.model = self.model.cuda()
        
        self.batch_size = args.batch_size
        self.num_epoch = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_length = args.warmup_length


        if args.lock_except_mlp:
            self.model.lock_except_mlp(unlocked_groups=args.lock_except_mlp_unlocked_groups)
        if args.lock_text:
            self.model.lock_text_tower()

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
           
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler =GradScaler()
    




    def train_epoch(self, train_loader, epoch, start_iter=0):

        num_batches_per_epoch = len(train_loader)
        for i, (images, texts, short_text) in enumerate(tqdm(train_loader, disable=(self.rank != 0))):
            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue
            images = images.cuda()
            texts = clipmoe.tokenize(texts, truncate=True).cuda()
            short_text = clipmoe.tokenize(short_text, truncate=True).cuda()
            self.scheduler(step)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss,loss_short = self.model(images, texts, short_text,self.rank)
                #alpha hyperparameter here
                alpha=1
                if self.args.use_short_text:
                    loss_total=loss+alpha*loss_short
                else:
                    loss_total=loss
            #self.optimizer.step()
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def train(self, exp_name,dataset,warmup_length=200):
        if 'share' in dataset:
            trainset =share4v_train_dataset()
            print('load sharegpt4v dataset, total samples:')
            print(len(trainset))
        elif 'datacomp' in dataset.lower():
            trainset = DCDataset()
            print('load datacomp dataset, total samples:')
            print(len(trainset))
        #init mcl data:
        if self.MCL_label_path is not None:
            cluster_labels=torch.load(self.MCL_label_path)
        else:
            cluster_labels=torch.zeros(len(trainset),dtype=torch.long)
        cluster_num=int(torch.max(cluster_labels))+1
        sampler=MCL_batch_sampler_DDP(cluster_labels,self.batch_size,cluster_num)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_sampler=sampler,
            num_workers=8,
            pin_memory=True,
        )

        
        self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length, steps=self.num_epoch * len(train_loader))
        start_epoch = 0
        resume_iter = 0
        for epoch in range(start_epoch, self.num_epoch):
            self.train_epoch(train_loader, epoch, start_iter=resume_iter)
            if self.rank == 0:
                torch.save(self.model.module.state_dict(), './checkpoints/'+exp_name+"_weights.pt")
                torch.save(self.optimizer.state_dict(), './checkpoints/'+exp_name+'_optimizer.pt')
                torch.save(self.scaler.state_dict(), './checkpoints/'+exp_name+'_scaler.pt')
        if self.num_epoch==0:
            if self.rank == 0:
                torch.save(self.model.module.state_dict(), './checkpoints/'+exp_name+"_weights.pt")
                torch.save(self.optimizer.state_dict(), './checkpoints/'+exp_name+'_optimizer.pt')
                torch.save(self.scaler.state_dict(), './checkpoints/'+exp_name+'_scaler.pt')

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()
    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_PORT"] = "29622"
        os.environ["MASTER_ADDR"] = addr
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    #torch.cuda.set_device(f'cuda:{rank % num_gpus}')
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(device=f'cuda:{rank % num_gpus}')
    return rank,rank % num_gpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--lr', default=1e-5, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=0.2, type=float, help='wd.')
    parser.add_argument('--log_scale', default=4.60517, type=float, help='clip temperature log scale.') #np.log(1/0.01)
    parser.add_argument("--exp-name", default="clipmoe", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")#200
    parser.add_argument("--base_model", default="ViT-L/14", help="CLIP Base Model, will load from OpenAI CLIP")
    parser.add_argument("--MCL-label-path",default=None, 
                        help="MCL pseudo label path. Set to None for normal training (without MCL negative sampling strategy)"
    )
    parser.add_argument('--dataset', type=str, default='datacomp', help='Dataset name. datacomp or sharegpt4v') 
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients. Useful for training vision tower only, when the text quality of the dataset is not good.",
    )
    parser.add_argument(
        "--lock-except-mlp",
        default=True,
        action='store_true',
        help="Lock all parameters except mlp layers.",
    )
    parser.add_argument(
        "--lock-except-mlp-unlocked-groups",
        type=int,
        default=0,
        help="Leave mlp groups unlocked from layer _ of the vision tower. Can be used to limit the final model size (i.e., control how many moe blocks are needed in the final model)",
    )
    parser.add_argument(
        "--use-short-text",
        default=False,
        action='store_true',
        help="whether to use the short text as regularization. Please refer to longclip. CLIPMoE does not need this by default.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size per gpu."
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train for."
    )

    args = parser.parse_args()
    rank,local_rank = setup_distributed()
    print("DDP Done")
   

    trainer = CLIP_Clean_Train(
        rank,
        local_rank, 
        args
    )
    trainer.train(resume=args.resume, warmup_length=args.warmup_length,dataset=args.dataset,exp_name=args.exp_name)
