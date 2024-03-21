"""
A much shorter version of train.py for benchmarking
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import functools
import torch
from model import GPTConfig, GPT, Block
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torch.distributed.fsdp.wrap import (
    _wrap_module_cls_individually
)


from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)

# -----------------------------------------------------------------------------
batch_size = 1
block_size = 1024 # how far back does the model look? i.e. context size
n_layer = 12
n_head = 12
n_embd = 768
dropout=False
bias = False
real_data = True
seed = 1337
backend = 'nccl'
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
profile = 'no' # "torch" or "cuda" profile or anything else - simple benchmark
# wandb logging
ddp = True
fsdp = False
wandb_log = False # disabled by default
wandb_project = 'benchmark'
wandb_run_name = 'run' + str(time.time())
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.dirname(__file__)+'/configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model init on CPU for fsdp
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=50304, dropout=dropout)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
Nparams = model.get_num_params()

print(f'Memory 1: {torch.cuda.memory_allocated()/1024/1024/1024} Gb, {torch.cuda.memory_reserved()/1024**3} Gb')


if ddp or fsdp:
    assert int(os.environ.get('RANK', -1)) != -1, "Not run with MPI"
    init_process_group(backend=backend)
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(ddp_local_rank)
    # device = f'{device}:{ddp_local_rank}'
master_process = os.environ.get('RANK', 0) == '0'

if not fsdp:
    model.to(device)
if ddp:
    model = DDP(model, gradient_as_bucket_view=True)

raw_model = model.module if ddp else model
if fsdp:
    model = FSDP(model,
        device_id=torch.cuda.current_device(),
        auto_wrap_policy=functools.partial(_wrap_module_cls_individually,
            module_classes={Block,}), #)
        use_orig_params=True)

print("TMP", raw_model.get_num_params(), Nparams)

print(f'Memory 2: {torch.cuda.memory_allocated()/1024/1024/1024} Gb, {torch.cuda.memory_reserved()/1024**3} Gb')


if compile:
    print("Compiling model...")
    model = torch.compile(model) # pytorch 2.0

# print(model)

optimizer = raw_model.configure_optimizers(model, weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)


# data loading init
if real_data:
    dataset = 'openwebtext'
    data_dir = os.path.join(os.path.dirname(__file__) + '/data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data # note ignore split in benchmarking script
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
else:
    # alternatively, if fixed data is desired to not care about data loading
    x = torch.randint(50304, (batch_size, block_size), device=device)
    y = torch.randint(50304, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)


# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)



def core_bench(stage, num_steps):
    torch.cuda.synchronize()
    t0 = time.time()
    X, Y = get_batch('train')
    for k in range(num_steps):
        with ctx:
            logits, loss = model(X, Y)
        X, Y = get_batch('train')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lossf = loss.item()
        print(f"{k}/{num_steps} loss: {lossf:.4f}")
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0
    mfu = raw_model.estimate_mfu(batch_size * 1 * num_steps, dt, N=Nparams)
    tpi = dt/num_steps
    m = torch.cuda.memory_allocated()/1024/1024/1024
    print(f"state = {stage}, loss: {lossf:.4f}, time per iteration: {tpi*1000:.4f}ms, "
        f"MFU: {mfu*100:.2f}%, MeM: {m} Gb, {torch.cuda.memory_reserved()/1024**3} Gb")

    if wandb_log and master_process:
        wandb.log({
            "time_per_iteration": tpi,
            "mfu": mfu*100, # convert to percentage
            "loss":lossf
        })

print(f'Memory 3: {torch.cuda.memory_allocated()/1024**3} Gb, {torch.cuda.memory_reserved()/1024**3} Gb')

if profile == "torch":
    # useful docs on pytorch profiler:
    # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 1, 1, 3
    num_stages = wait + warmup + active
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=False,
        profile_memory=False,
        with_stack=False, # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False, # only for torchscript models atm
    ) as prof:
        for stage, num_steps in enumerate([10] * num_stages):
            core_bench(0, num_steps)
            prof.step() # notify the profiler at end of each step
elif profile == "cuda":
    with torch.cuda.profiler.profile():
        with torch.autograd.profiler.emit_nvtx():
            for stage, num_steps in enumerate([4, 10]): # burnin, then benchmark
                core_bench(stage, num_steps)


else:
    for stage, num_steps in enumerate([2] * 5): # burnin, then benchmark
        core_bench(stage, num_steps)

if ddp:
    destroy_process_group()
