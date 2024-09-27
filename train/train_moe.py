from cgitb import text
import torch
from utils import concat_all_gather, is_dist_avail_and_initialized, accuracy
import torch.distributed as dist
from tqdm import tqdm
from dataset import share4v_train_dataset,DCDataset
from clipmoe import clipmoe
from torch.utils.data.distributed import DistributedSampler
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
        self.rank=rank
        self.local_rank = local_rank
        if args.init=="MCL":
            checkpoints=['ckp0',
                'ckp1',
                'ckp2',
                'ckp3']      
            #init a clip-moe from the ckps list
            self.model, _ = clipmoe.initMoE_MCL(checkpoints,top_k=args.top_k,moe_layers=args.moe_layers,dropout=0.1, device='cpu')
        else:
            #init a clip-moe from the dense checkpoint by sparse upcycling
            checkpoint='ckp0'
            self.model, _=clipmoe.initMoE_upcycle(checkpoint,top_k=args.top_k,num_experts=4,dropout=0.1, moe_layers=args.moe_layers,device='cpu')
        self.model.train()
        self.model.use_short_text=args.use_short_text
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)  
        self.model = self.model.cuda()
        
        self.batch_size = args.batch_size
        self.num_epoch = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_length = args.warmup_length
        
        if args.lock_except_gate:
            self.model.lock_except_gate()

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler =GradScaler()


    def train_epoch(self, train_loader, epoch):
        num_batches_per_epoch = len(train_loader)
        for i, (images, texts, short_text) in enumerate(tqdm(train_loader, disable=(self.rank != 0))):
            step = num_batches_per_epoch * epoch + i
            images = images.cuda()
            texts = clipmoe.tokenize(texts, truncate=True).cuda()
            short_text = clipmoe.tokenize(short_text, truncate=True).cuda()
            self.scheduler(step)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss,loss_short,loss_router_image,loss_router_textl,loss_router_texts = self.model(images, texts, short_text,self.rank)
                #alpha hyperparameter here
                alpha=1
                beta=0.01
                loss_total=loss+alpha*loss_short+beta*(loss_router_image+loss_router_textl/2+loss_router_texts/2)
            #self.optimizer.step()
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
           

    def train(self, dataset,exp_name,warmup_length=200):
    
        #init data:
        if 'share' in dataset:
            trainset =share4v_train_dataset()
            print('load sharegpt4v dataset, total samples:')
            print(len(trainset))
        elif 'datacomp' in dataset.lower():
            trainset = DCDataset()
            print('load datacomp dataset, total samples:')
            print(len(trainset))
        train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=6, pin_memory=True)
        self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length, steps=self.num_epoch * len(train_loader))
        start_epoch = 0
        resume_iter = 0
        for epoch in range(start_epoch, self.num_epoch):
            self.train_epoch(train_loader, epoch, start_iter=resume_iter)
            if self.rank == 0:
                torch.save(self.model.module.state_dict(), './checkpoints/'+exp_name+'_weights.pt')

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
        os.environ["MASTER_PORT"] = "29522"
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
    parser.add_argument('--lr', default=1e-6, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='wd.')
    parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')#log(1/0.01)=4.6052 #0.69
    parser.add_argument("--exp-name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")#200
    parser.add_argument('--dataset', type=str, default='datacomp', help='Dataset name.  datacomp or sharegpt4v') #datacomp, sharegpt4v
    parser.add_argument(
        "--unlock-transformer-blocks",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked."
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size per gpu."
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--use-short-text",
        default=False,
        action='store_true',
        help="whether to use the short text as regularization.Please refer to longclip. CLIPMoE does not need this by default.",
    )
    #MoE params
    parser.add_argument(
        "--top-k", type=int, default=2, help="top_k for MoE"
    )
    parser.add_argument(
        "--moe-layers", type=int, default=24, help="num of MoE layers. should fit with --lock-except-mlp-unlocked-groups in train_mcl.py"
    )
    parser.add_argument("--init", default="upcycling", type=str, help="specify init methods (MCL, upcycling).")
    parser.add_argument(
        "--lock-except-gate",
        default=True,
        action='store_true',
        help="Lock all parameters except MoE router.",
    )

    args = parser.parse_args()
    rank,local_rank = setup_distributed()
    print("DDP Done")
   

    trainer = CLIP_Clean_Train(
        rank,
        local_rank, 
        args
    )
    trainer.train(resume=args.resume, warmup_length=args.warmup_length,dataset=args.dataset)
