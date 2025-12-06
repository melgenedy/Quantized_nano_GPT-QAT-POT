"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os


# Redirect Hugging Face caches to a larger disk
os.environ["HF_HOME"] = "/local/mnt/workspace/melgened/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/local/mnt/workspace/melgened/hf_home/transformers"
os.environ["HF_DATASETS_CACHE"] = "/local/mnt/workspace/melgened/hf_home/datasets"

# Ensure directories exist
for p in [os.environ["HF_HOME"], os.environ["TRANSFORMERS_CACHE"], os.environ["HF_DATASETS_CACHE"]]:
    os.makedirs(p, exist_ok=True)

import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
import copy


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 64 #1024
# model
n_layer = 4 #12
n_head = 4 #12
n_embd = 128 # 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 30000 #600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 30000 #600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cpu' #'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False #True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(file='config/finetune_openwebQAT.py').read()) #exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
#from config import train_shakespeare_char

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    elif split == 'val':
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'test.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

#init_from = 'resume'    

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt_qt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

model_base = copy.deepcopy(model)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    #model.eval()
    move_exported_model_to_eval(model)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                #logits, loss = model(X, Y)
                logits = model(X)
                logits_=(logits[0]).squeeze(1)
                targets_last = Y[:, -1]
                loss = torch.nn.functional.cross_entropy(logits_, targets_last)
            losses[k] = loss.item()
        out[split] = losses.mean()
    #model.train()
    move_exported_model_to_train(model)
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    #aa9115eb29b93c6aab8111d29d59ad1da6f4e606

def get_batch_for_test(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    elif split == 'val':
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'test.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

##############################################################################################################
'''
Adding QAT to the model

'''

'''
Configure Quantization specs, and define get_nanoGPT_quantization_config that will trigger the PoT fake quantize (FakeQuantize_PoT) instead of normal fake quantize

'''

activations_dtype = torch.int32
weights_dtype = torch.int32

is_pot = True

import functools

from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
)
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.utils import _get_module_name_filter
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    _convert_scalars_to_attrs,
    OP_TO_ANNOTATOR,
    OperatorConfig,
    OperatorPatternType,
    propagate_annotation,
    QuantizationConfig,
)
if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
    from torch.fx import Node

from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e, prepare_qat_pt2e
from torch.ao.quantization import MovingAverageMinMaxObserver
from torch.ao.quantization.quantizer.xnnpack_quantizer import (XNNPACKQuantizer,get_symmetric_quantization_config,)
# 
from torch.ao.quantization import move_exported_model_to_train, move_exported_model_to_eval
from fakequant_pot import FakeQuantize_PoT

from bert_utils import compute_bert_compare

@functools.lru_cache
def get_nanoGPT_quantization_config(
    is_per_channel: bool = False,
    is_qat: bool = False,
    is_dynamic: bool = False,
    act_qmin: int = -2**31,#-128,#-2**31,#-128, # activation is kept at high precision, but can be lowered to 16 bits without noticeable loss
    act_qmax: int = 2**31-1,#127,#2**31-1,#127,
    # following are the two parameter we can change
    weight_qmin: int = -2**7,#-8,#-127,#-15,#-2**31+1,#-127,
    weight_qmax: int = 2**7,#7,#127#15,#2**31-1,#127,
):
    extra_args: Dict[str, Any] = {"eps": 2**-12}        # 2**-12
    if 0:#is_qat:  #don't use qat for activations
        if is_dynamic:
            act_observer_or_fake_quant_ctr = FakeQuantize_PoT
            dynamic_quant_observer = MovingAverageMinMaxObserver.with_args(
                averaging_constant=1
            )
            extra_args["observer"] = dynamic_quant_observer
        else:
            act_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize  # type: ignore[assignment]
    else:
        if is_dynamic:
            act_observer_or_fake_quant_ctr = PlaceholderObserver  # type: ignore[assignment]
        else:
            act_observer_or_fake_quant_ctr = HistogramObserver  # type: ignore[assignment]

    act_quantization_spec = QuantizationSpec(
        dtype=activations_dtype,#int8,
        quant_min=act_qmin,
        quant_max=act_qmax,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
            **extra_args,
        ),
    )
    weight_qscheme = (
        torch.per_channel_symmetric if is_per_channel else torch.per_tensor_symmetric
    )
    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        MinMaxObserver
    )
    if is_qat:
        # TODO: qat + per channel?
        if is_pot:
            weight_observer_or_fake_quant_ctr = FakeQuantize_PoT
            dynamic_quant_observer = MovingAverageMinMaxObserver.with_args(
                averaging_constant=1
            )
            extra_args["observer"] = dynamic_quant_observer        
        else:
            weight_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize
    elif is_per_channel:
        weight_observer_or_fake_quant_ctr = PerChannelMinMaxObserver

    extra_args: Dict[str, Any] = {"eps": 2**-5}
    if is_qat:
        if weight_qscheme == torch.per_tensor_symmetric:
            extra_args["observer"] = MovingAverageMinMaxObserver
        else:
            extra_args["observer"] = MovingAveragePerChannelMinMaxObserver  # type: ignore[dict-item]
    weight_quantization_spec = QuantizationSpec(
        dtype=weights_dtype,#int8,
        quant_min=weight_qmin,
        quant_max=weight_qmax,
        qscheme=weight_qscheme,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(
            **extra_args
        ),
    )

    bias_quantization_spec = None
    if is_dynamic:
        quantization_config = QuantizationConfig(
            act_quantization_spec,
            None,
            weight_quantization_spec,
            bias_quantization_spec,
            is_qat,
        )
    else:
        quantization_config = QuantizationConfig(
            act_quantization_spec,
            act_quantization_spec,
            weight_quantization_spec,
            bias_quantization_spec,
            is_qat,
        )
    return quantization_config

###################################################################################################
'''Export and prepare model for QAT'''

X, Y = get_batch('train') # fetch the very first batch
#model.qconfig = torch.quantization.default_qconfig
mod=torch.export.export_for_training(model,(X,)).module()

#quantizedquantizer2 = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_qat=False))
quantizedquantizer2 = XNNPACKQuantizer().set_global(get_nanoGPT_quantization_config(is_qat=True,is_dynamic=False))

model=prepare_qat_pt2e(mod, quantizedquantizer2)

for iter in range(100):
    X, Y = get_batch_for_test('val')
    model(X)

m_quant = copy.deepcopy(model)
m_quant = convert_pt2e(m_quant)    
####################################################################################################

'''Continue with normal training loop'''


data=[]
# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        data.append([iter_num, losses['train'], losses['val']])
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                mod_q = copy.deepcopy(model)
                mod_quant = convert_pt2e(mod_q)
                checkpoint = {
                    'model': mod_quant.state_dict(),
                    #'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt_qt.pt'))
                bert_base, bert_quant = compute_bert_compare(model_base,model, data_dir)
                print(f"bert base {bert_base}, bert quant {bert_quant}")
                bert_diff = bert_base - bert_quant
                bert_diff_abs = np.abs(bert_diff)
                if bert_diff_abs < .02:
                    []
                if wandb_log:
                    wandb.log({
                        "bert_base": bert_base,
                        "bert_quant": bert_quant,
                        "bert diff": bert_diff,
                        "bert_diff_abs":bert_diff_abs,
                    })

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            #logits, loss = model(X, Y)
            logits = model(X)
            logits_=(logits[0]).squeeze(1)
            targets_last = Y[:, -1]
            loss = torch.nn.functional.cross_entropy(logits_, targets_last)            
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = 1#raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = 1#mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()


#################################################################################################################################
# print out training results
from tabulate import tabulate
headers = ["Step", "Train Loss", "Val Loss"]
print(tabulate(data, headers=headers, tablefmt="grid"))


##############################################################################################################


################################################################################
'''PTQ model'''
import math
# Function to round to the nearest power of 2
def round_to_power_of_2(x):
    abs_x = abs(x)
    power = torch.round(torch.log2(abs_x))
    #if abs_x == 0:
    #  return 0
    power[abs_x==0] = 0
    #power[power == 15] = 14
    y = 2**power * torch.sign(x)
    y[abs_x==0] = 0
    #if 2**power < abs_x:
    #    power +=1
    #if power >=8:
    #    power=7
    return y#2**power * torch.sign(x)#(1 if x > 0 else -1)

quantized_weights = m_quant.state_dict()
for name, param in quantized_weights.items():
    if isinstance(param, torch.Tensor) and param.dtype == (weights_dtype):
        # Convert to float, round to power of 2, convert back to quantized
        float_param = param.dequantize()
        #rounded_param = torch.tensor([round_to_power_of_2(x) for x in float_param.flatten()], dtype=torch.float32).reshape(param.shape).to(torch.int8)
        rounded_param = torch.tensor(round_to_power_of_2(float_param.flatten()), dtype=torch.float32).reshape(param.shape).to(weights_dtype)
        
        #rounded_param = torch.tensor([round_to_power_of_2(x) for x in float_param.flatten()], dtype=torch.float32).reshape(param.shape)
        #quantized_param = torch.quantize_per_tensor(rounded_param, 1.0, 0, torch.qint8)
        print(name, 'max', torch.max(float_param), 'min', torch.min(float_param))
        print(name, 'max', torch.max(rounded_param), 'min', torch.min(rounded_param))
        
        # Update the model's state_dict with the modified weights
        quantized_weights[name] = rounded_param


m_quant.load_state_dict(quantized_weights)
#########################################################################

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss_testing(model_test):
    out = {}
    #model.eval()
    move_exported_model_to_eval(model_test)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                #logits, loss = model(X, Y)
                logits = model_test(X)
                logits_=(logits[0]).squeeze(1)
                targets_last = Y[:, -1]
                loss = torch.nn.functional.cross_entropy(logits_, targets_last)
            losses[k] = loss.item()
        out[split] = losses.mean()
    #model.train()
    move_exported_model_to_train(model_test)
    return out
##########################################################################


import tiktoken

# ok let's assume gpt-2 encodings by default
#print("No meta.pkl found, assuming GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(tokens=l)


def generate(Model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= 64 else idx[:, -64:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = Model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        
        import torch.nn.functional as F
        
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# model_base.eval()
# move_exported_model_to_eval(model)
# #move_exported_model_to_eval(mod_q)
# move_exported_model_to_eval(mod_quant)
# move_exported_model_to_eval(m_quant)

################################################################################################

def get_batch_for_bert_test(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    batch_size_test = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size_test = 128 #1024    

    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    elif split == 'val':
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'test.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size_test, (batch_size_test,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size_test]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size_test]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

#################################################################################################


import evaluate

bertscore = evaluate.load("bertscore")

def compute_bert(refs, cands):
    results = bertscore.compute(
        predictions=cands,
        references=refs,
        lang="en",                  # set language
        model_type="roberta-base", # optional; default is roberta-large
        rescale_with_baseline=True
    )
    return results

max_new_tokens = 64 #500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
n_iters = 10
baseline_avgF1 = np.zeros(10)
qat_convert_avgF1 = np.zeros(10)
qat_latest_avgF1 = np.zeros(10)
ptq_avgF1 = np.zeros(10)

with torch.no_grad():

    model_base.eval()
    move_exported_model_to_eval(model)
    #move_exported_model_to_eval(mod_q)
    #move_exported_model_to_eval(mod_quant)
    #move_exported_model_to_eval(m_quant)        
    with ctx:
        for k in range(n_iters):

            X_test, y = get_batch_for_bert_test('val')
            X = X_test[:,:64]
            X_ref = X_test[:,64:]
            
            print(decode(X[0].tolist()),end="")
            y_org = model_base.generate(X, max_new_tokens, temperature=temperature, top_k=top_k)
            #print(decode(y_org[0].tolist()),end="")
            #print('\n---------------')

            y_last = generate(model, X, max_new_tokens, temperature=temperature, top_k=top_k)
            #print(decode(y_last[0].tolist()),end="")
            #print('\n---------------')

            #y_best = generate(mod_q, X, max_new_tokens, temperature=temperature, top_k=top_k)
            #print(decode(y_best[0].tolist()),end="")
            #print('\n---------------')

            #y_convert = generate(mod_quant, X, max_new_tokens, temperature=temperature, top_k=top_k)
            #print(decode(y_convert[0].tolist()),end="")
            #print('\n---------------')            

            #y_ptq = generate(m_quant, X, max_new_tokens, temperature=temperature, top_k=top_k)
        # print(decode(y_ptq[0].tolist()),end="")
            #print('\n---------------') 

            
            # Calculate BLEU scores
            reference_texts = [decode(X_test[0,64:64+64].tolist())]
            baseline_outputs = [decode(y_org[0,64:64+64].tolist())]
            #qat_convert_outputs = [decode(y_convert[0,64:64+64].tolist())]
            qat_latest_outputs = [decode(y_last[0,64:64+64].tolist())]
            #ptq_convert_outputs = [decode(y_ptq[0,64:64+64].tolist())]

            results = compute_bert(reference_texts, baseline_outputs)

            #print("Per-sentence F1:", [round(f, 4) for f in results["f1"]])
            print("Average F1 - Base:", sum(results["f1"]) / len(results["f1"]))

            baseline_avgF1[k] = sum(results["f1"]) / len(results["f1"])

            #results = compute_bert(reference_texts, qat_convert_outputs)

            #print("Per-sentence F1:", [round(f, 4) for f in results["f1"]])
            #print("Average F1 - qat convert:", sum(results["f1"]) / len(results["f1"]))

            #qat_convert_avgF1[k] = sum(results["f1"]) / len(results["f1"])

            results = compute_bert(reference_texts, qat_latest_outputs)

            #print("Per-sentence F1:", [round(f, 4) for f in results["f1"]])
            print("Average F1 - qat before convert:", sum(results["f1"]) / len(results["f1"]))

            qat_latest_avgF1[k] = sum(results["f1"]) / len(results["f1"])

            #results = compute_bert(reference_texts, ptq_convert_outputs)

            #print("Per-sentence F1:", [round(f, 4) for f in results["f1"]])
            #print("Average F1 - ptq:", sum(results["f1"]) / len(results["f1"]))   

            #ptq_avgF1[k] = sum(results["f1"]) / len(results["f1"])

    model_base.train()
    move_exported_model_to_train(model)
    #move_exported_model_to_eval(mod_q)
    #move_exported_model_to_train(mod_quant)
    #move_exported_model_to_train(m_quant)
    
#baseline_avgF1, qat_convert_avgF1, qat_latest_avgF1, ptq_avgF1 = compute_bert_compare(model_base, model, mod_quant, m_quant)