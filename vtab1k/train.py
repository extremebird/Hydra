from torch.optim import AdamW
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from dataset import *
from utils import *
from hydra import set_hydra
from torch import nn
from timm.data import Mixup
from timm.loss import  SoftTargetCrossEntropy

def train(config, model, dl, opt, scheduler, epoch,criterion=nn.CrossEntropyLoss()):
    model.train()
    model = model.cuda()
    for ep in tqdm(range(epoch)):
        model.train()
        model = model.cuda()
        # pbar = tqdm(dl)
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = criterion(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 10 == 9:
            acc = test(model, test_dl)
            if acc > config['best_acc']:
                config['best_acc'] = acc
                save(config['method'], config['name'], model, acc, ep)
    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    #pbar = tqdm(dl)
    model = model.cuda()
    for batch in dl:  # pbar:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y)

    return acc.result()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=-1.0)
    parser.add_argument('--wd', type=float, default=-1.0)
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--method', type=str, default='hydra_both',choices=['hydra_both'])
    parser.add_argument('--scale', type=float, default=0)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--dropout',   type=float, default=-1.0)
    parser.add_argument('--bsize',   type=int, default=-1)
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    config = get_config(args.method, args.dataset)
    config = set_hyperparameter(config, args)

    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./models/%s'%(args.method)):
        os.makedirs('./models/%s'%(args.method))


    model = create_model('vit_base_patch16_224_in21k', drop_path_rate=0.1,checkpoint_path='./ViT-B_16.npz')

    model.cuda()
    throughput(model)
    train_dl, test_dl = get_data(args.dataset, batch_size=config['batchsize'])
    print(f'train set: {len(train_dl)}, test set: {len(test_dl)}')
    
    set_hydra(model, args.method, dim=args.dim, configs=config)
    model.cuda()
    throughput(model)

    if hasattr(model,'blocks'):
        print(model.blocks[0])
    elif hasattr(model,'layers'):
        print(model.layers[0])
    elif hasattr(model,'stages'):
        print(model.stages[0])
    else:
        assert NotImplementedError

    trainable = []
    model.reset_classifier(config['class_num'])
    
    config['best_acc'] = 0
    config['method'] = args.method
    total=0
    if 'hydra' in args.method:
        key_trainable = 'hydra'

    for n, p in model.named_parameters():
        if key_trainable in n or 'head' in n or 'scale' in n:
            trainable.append(p)
            total+=p.nelement()
        else:
            p.requires_grad = False
    print('  + Number of trainable params: %.2fK' % (total / 1e3))
    opt = AdamW(trainable, lr=config['lr'], weight_decay=config['wd'])
    scheduler = CosineLRScheduler(opt, t_initial=500,
                                  warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, cycle_decay=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    model = train(config, model, train_dl, opt, scheduler, epoch=100,criterion=criterion)
    print(config['best_acc'])
    print('complete')
    for n, p in model.named_parameters():
        if 'scale' in n:
            print(n, p)

    