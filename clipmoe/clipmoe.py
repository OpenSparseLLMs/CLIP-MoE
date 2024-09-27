import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from pkg_resources import packaging
from torch import nn
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model_clipmoe import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["load", "tokenize"]
_tokenizer = _Tokenizer()


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])



def load(model_path: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None):
    state_dict = torch.load(model_path, map_location="cpu")
    model = build_model(state_dict or model.state_dict(), load_from_clip = False).to(device)
    if str(device) == "cpu":
        model.float()
    return model, _transform(model.visual.input_resolution)

def loadMoE(model_path, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None,top_k=0,num_experts=0,dropout=0,moe_layers=24):
    '''
    load CLIP MoE from a checkpoint
    '''
   
    state_dict = torch.load(model_path, map_location="cpu")

 
    model = build_model(state_dict, load_from_clip = False,MoE_args=[num_experts,top_k,dropout,moe_layers]).to(device)
    #load parameters
    model.load_state_dict(state_dict)
    if str(device) == "cpu":
        model.float()

    return model.to(device), _transform(model.visual.input_resolution)

def initMoE_MCL(model_path_list, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None,top_k=0,moe_layers=24,dropout=0):
    '''
    initialize a CLIP-MoE from a list of CLIP parameters.
    the list of CLIP parameters only differ in MLP
    the experts (mlps) of CLIP-MoE are from the list of CLIP mlp parameters respectively
    other parameters except for mlp are the same as CLIP parameters
    '''
    state_dicts=[]
    for model_path in model_path_list:
        state_dict = torch.load(model_path, map_location="cpu")
        state_dicts.append(state_dict)
    num_experts=len(state_dicts)
 
    model = build_model(state_dicts[0], load_from_clip = False,MoE_args=[num_experts,top_k,dropout,moe_layers]).to(device)
    #load parameters except for mlps
    general_state_dict = {k: v for k, v in state_dicts[0].items() if not ('mlp' in k)}
    model.load_state_dict(general_state_dict, strict=False)
    #load parameters for experts
    for i in range(num_experts):
        dense_layers=len(model.visual.transformer.resblocks)-moe_layers
        if i==0:
            blockid=0
            for resblock in model.visual.transformer.resblocks[:dense_layers]:
                resblock.mlp.c_fc.weight.data.copy_(state_dicts[i]['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_fc.weight'])
                resblock.mlp.c_fc.bias.data.copy_(state_dicts[i]['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_fc.bias'])
                resblock.mlp.c_proj.weight.data.copy_(state_dicts[i]['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_proj.weight'])
                resblock.mlp.c_proj.bias.data.copy_(state_dicts[i]['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_proj.bias'])
                blockid+=1
        blockid=dense_layers
        for resblock in model.visual.transformer.resblocks[dense_layers:]:
            resblock.experts[i].c_fc.weight.data.copy_(state_dicts[i]['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_fc.weight'])
            resblock.experts[i].c_fc.bias.data.copy_(state_dicts[i]['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_fc.bias'])
            resblock.experts[i].c_proj.weight.data.copy_(state_dicts[i]['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_proj.weight'])
            resblock.experts[i].c_proj.bias.data.copy_(state_dicts[i]['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_proj.bias'])
            blockid+=1
        blockid=0
        for resblock in model.transformer.resblocks:
            resblock.experts[i].c_fc.weight.data.copy_(state_dicts[i]['transformer.resblocks.'+str(blockid)+'.mlp.c_fc.weight'])
            resblock.experts[i].c_fc.bias.data.copy_(state_dicts[i]['transformer.resblocks.'+str(blockid)+'.mlp.c_fc.bias'])
            resblock.experts[i].c_proj.weight.data.copy_(state_dicts[i]['transformer.resblocks.'+str(blockid)+'.mlp.c_proj.weight'])
            resblock.experts[i].c_proj.bias.data.copy_(state_dicts[i]['transformer.resblocks.'+str(blockid)+'.mlp.c_proj.bias'])
            blockid+=1    
        
    if str(device) == "cpu":
        model.float()

    return model.to(device), _transform(model.visual.input_resolution)

def initMoE_upcycle(model_path, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None,top_k=0,num_experts=0,dropout=0,moe_layers=24):
    '''
    initialize a CLIP-MoE from a CLIP model by upscaling the mlp module.
    '''
   
    state_dict = torch.load(model_path, map_location="cpu")

 
    model = build_model(state_dict, load_from_clip = False,MoE_args=[num_experts,top_k,dropout,moe_layers]).to(device)
    #load parameters except for mlps
    general_state_dict = {k: v for k, v in state_dict.items() if not ('mlp' in k)}
    model.load_state_dict(general_state_dict, strict=False)
    #load parameters for experts
    for i in range(num_experts):
        dense_layers=len(model.visual.transformer.resblocks)-moe_layers
        if i==0:
            blockid=0
            for resblock in model.visual.transformer.resblocks[:dense_layers]:
                resblock.mlp.c_fc.weight.data.copy_(state_dict['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_fc.weight'])
                resblock.mlp.c_fc.bias.data.copy_(state_dict['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_fc.bias'])
                resblock.mlp.c_proj.weight.data.copy_(state_dict['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_proj.weight'])
                resblock.mlp.c_proj.bias.data.copy_(state_dict['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_proj.bias'])
                blockid+=1
        blockid=dense_layers
        for resblock in model.visual.transformer.resblocks[dense_layers:]:
            resblock.experts[i].c_fc.weight.data.copy_(state_dict['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_fc.weight'])
            resblock.experts[i].c_fc.bias.data.copy_(state_dict['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_fc.bias'])
            resblock.experts[i].c_proj.weight.data.copy_(state_dict['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_proj.weight'])
            resblock.experts[i].c_proj.bias.data.copy_(state_dict['visual.transformer.resblocks.'+str(blockid)+'.mlp.c_proj.bias'])
            blockid+=1
        
        dense_layers=len(model.transformer.resblocks)-moe_layers
        if dense_layers<0: 
            dense_layers=0
        if i==0:
            blockid=0
            for resblock in model.transformer.resblocks[:dense_layers]:
                resblock.mlp.c_fc.weight.data.copy_(state_dict['transformer.resblocks.'+str(blockid)+'.mlp.c_fc.weight'])
                resblock.mlp.c_fc.bias.data.copy_(state_dict['transformer.resblocks.'+str(blockid)+'.mlp.c_fc.bias'])
                resblock.mlp.c_proj.weight.data.copy_(state_dict['transformer.resblocks.'+str(blockid)+'.mlp.c_proj.weight'])
                resblock.mlp.c_proj.bias.data.copy_(state_dict['transformer.resblocks.'+str(blockid)+'.mlp.c_proj.bias'])
                blockid+=1
        blockid=dense_layers
        for resblock in model.transformer.resblocks[dense_layers:]:
            resblock.experts[i].c_fc.weight.data.copy_(state_dict['transformer.resblocks.'+str(blockid)+'.mlp.c_fc.weight'])
            resblock.experts[i].c_fc.bias.data.copy_(state_dict['transformer.resblocks.'+str(blockid)+'.mlp.c_fc.bias'])
            resblock.experts[i].c_proj.weight.data.copy_(state_dict['transformer.resblocks.'+str(blockid)+'.mlp.c_proj.weight'])
            resblock.experts[i].c_proj.bias.data.copy_(state_dict['transformer.resblocks.'+str(blockid)+'.mlp.c_proj.bias'])
            blockid+=1    
        
    if str(device) == "cpu":
        model.float()

    return model.to(device), _transform(model.visual.input_resolution)
        
    

    def _node_get(node: torch._C.Node, key: str):
        """Gets attributes of a node which is polymorphic over return type.
        
        From https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def load_from_clip(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """Load from CLIP model for fine-tuning 

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """

    _MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
    }

    def available_models() -> List[str]:
        """Returns the names of available CLIP models"""
        return list(_MODELS.keys())

    def _download(url: str, root: str):
        os.makedirs(root, exist_ok=True)
        filename = os.path.basename(url)

        expected_sha256 = url.split("/")[-2]
        download_target = os.path.join(root, filename)

        if os.path.exists(download_target) and not os.path.isfile(download_target):
            raise RuntimeError(f"{download_target} exists and is not a regular file")

        if os.path.isfile(download_target):
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

        with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
            raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

        return download_target

    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    model = build_model(state_dict or model.state_dict(), load_from_clip = True).to(device)
        
    positional_embedding_pre = model.positional_embedding.type(model.dtype)
            
    length, dim = positional_embedding_pre.shape
    keep_len = 20
    posisitonal_embedding_new = torch.zeros([4*length-3*keep_len, dim], dtype=model.dtype)
    for i in range(keep_len):
        posisitonal_embedding_new[i] = positional_embedding_pre[i]
    for i in range(length-1-keep_len):
        posisitonal_embedding_new[4*i + keep_len] = positional_embedding_pre[i + keep_len]
        posisitonal_embedding_new[4*i + 1+keep_len] = 3*positional_embedding_pre[i + keep_len]/4 + 1*positional_embedding_pre[i+1+keep_len]/4
        posisitonal_embedding_new[4*i + 2+keep_len] = 2*positional_embedding_pre[i+keep_len]/4 + 2*positional_embedding_pre[i+1+keep_len]/4
        posisitonal_embedding_new[4*i + 3+keep_len] = 1*positional_embedding_pre[i+keep_len]/4 + 3*positional_embedding_pre[i+1+keep_len]/4

    posisitonal_embedding_new[4*length -3*keep_len - 4] = positional_embedding_pre[length-1] + 0*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
    posisitonal_embedding_new[4*length -3*keep_len - 3] = positional_embedding_pre[length-1] + 1*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
    posisitonal_embedding_new[4*length -3*keep_len - 2] = positional_embedding_pre[length-1] + 2*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
    posisitonal_embedding_new[4*length -3*keep_len - 1] = positional_embedding_pre[length-1] + 3*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
            
    positional_embedding_res = posisitonal_embedding_new.clone()
            
    model.positional_embedding = nn.Parameter(posisitonal_embedding_new, requires_grad=False)
    model.positional_embedding_res = nn.Parameter(positional_embedding_res, requires_grad=True)

    if str(device) == "cpu":
        model.float()
    return model, _transform(model.visual.input_resolution)
        
    def _node_get(node: torch._C.Node, key: str):
        """Gets attributes of a node which is polymorphic over return type.
        
        From https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())

def tokenize(texts: Union[str, List[str]], context_length: int = 77*4-60, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
