
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
from timm.models import create_model
from argparse import ArgumentParser
from dataset import *
from utils import *
from hydra import set_hydra, set_hydraWeight


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    pbar = tqdm(dl)
    model = model.cuda()
    for batch in pbar:  # pbar:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y)

    return acc.result()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=-1)
    parser.add_argument('--wd', type=float, default=-1)
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--method', type=str, default='hydra_both', choices=['hydra_ffn', 'hydra_both'])
    parser.add_argument('--scale', type=float, default=0)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--bsize', type=int, default=-1)
    parser.add_argument('--dropout', type=float, default=-1)
    args = parser.parse_args()
    print(args)
    config = get_config(args.method, args.dataset)
    config = set_hyperparameter(config, args)

    # build model
    model = create_model('vit_base_patch16_224_in21k', drop_path_rate=0.1, checkpoint_path='./ViT-B_16.npz')

    # build dataset
    train_dl, test_dl = get_data(args.dataset)

    # running throughput
    model.cuda()
    print('before reparameterizing: ')
    throughput(model)

    # build hydra
    set_hydra(model, args.method, configs=config, dim=args.dim, set_forward=False)
    print('Add Hydra branch: ')
    throughput(model)

    # load model
    model.reset_classifier(config['class_num'])
    model = load(args.method, config['name'], model)

    set_hydraWeight(model, args.method, config, dim=args.dim)

    # running throughput
    model.cuda()
    print()
    print('after reparameterizing: ')
    throughput(model)

    # testing loop
    acc = test(model, test_dl)
    print('Accuracy:', acc)

