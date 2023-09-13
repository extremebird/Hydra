import random
import yaml
import time
import numpy as np
import torch

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def throughput(model,img_size=224,bs=1):
    with torch.no_grad():
        x = torch.randn(bs, 3, img_size, img_size).cuda()
        batch_size=x.shape[0]
        model.eval()
        for i in range(50):
            model(x)
        torch.cuda.synchronize()
        print(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(x)
        torch.cuda.synchronize()
        tic2 = time.time()
        print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        MB = 1024.0 * 1024.0
        print('memory:', torch.cuda.max_memory_allocated() / MB)

@torch.no_grad()
def save(method, dataset, model, acc, ep):
    model.eval()
    model = model.cpu()
    trainable = {}
    if 'hydra' in method:
        train_key = 'hydra'
    else:
        train_key = 'adapter'
    for n, p in model.named_parameters():
        if train_key in n or 'head' in n:
            trainable[n] = p.data
    torch.save(trainable, './models/%s/%s.pt'%(method, dataset))
    with open('./models/%s/%s.log'%(method, dataset), 'w') as f:
        f.write(str(ep)+' '+str(acc))
        

def load(method, dataset, model):
    model = model.cpu()
    st = torch.load('./models/%s/%s.pt'%(method, dataset))
    model.load_state_dict(st, False)
    return model

def get_config(method, dataset_name):
    config_name = './configs/%s/%s.yaml' % (method, dataset_name)
    with open(config_name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def set_hyperparameter(config, args):
    if args.scale != 0:
        config['scale'] = args.scale
    if args.bsize > 0: 
        config['batchsize'] = args.bsize
    if args.lr > 0:
        config['lr'] = args.lr
    if args.wd >= 0:
        config['wd'] = args.wd
    if args.dropout >= 0:
        config['dropout'] = args.dropout
    return config