import argparse
from clipmoe import clipmoe
import torch
from tqdm import tqdm
from classes import imagenet_classes
from data_loader import data_loader
from templates import imagenet_templates

def get_label(fold_name):
    return torch.tensor([int(fold_name)])
def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clipmoe.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clipmoe.loadMoE('checkpoint',top_k=2,num_experts=4,moe_layers=24, device=device)   

    model.to(device)
    softmax = torch.nn.Softmax(dim=1)
    loader, dataset = data_loader(preprocess, args)
    model.eval()
    
    zeroshot_weights = zeroshot_classifier(model, imagenet_classes, imagenet_templates)
    total_num = 0
    true_num = 0
    total_targets = torch.zeros(1000)
    with torch.no_grad():
        for i, (images, targets, paths) in enumerate(dataset):
            #print(targets)
            total_targets[get_label(targets)] = 1
        total_targets = total_targets.to(device)
        for i, (images, targets, paths) in enumerate(tqdm(loader)):
            images = images.to(device)

            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights
            logits = logits * total_targets
            logits = softmax(logits)
            pred = torch.argmax(logits,dim=1)
            
            total_len = pred.shape[0]
            for i in range(total_len):
                label = targets[i]
                label = get_label(label).item()
                if pred[i].item() == label:
                    true_num += 1
                total_num += 1
            
            #save_to_file(logits, targets, paths)
    print(true_num / total_num)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='CLIP inference')
    args.add_argument('-d', '--data-dir', default='imagenet-v2-data', type=str,
                      help='dataset path (default: None)')
    args.add_argument('-w', '--num-workers', default=16, type=int,
                      help='number of workers (default: 64)')
    args.add_argument('-b', '--batch_size', default=1024, type=int,
                      help='Batch size (default: 64)')

    config = args.parse_args()
    main(config)
