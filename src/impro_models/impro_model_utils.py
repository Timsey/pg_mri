import torch
import shutil

from .convpool_model import build_impro_convpool_model
from .convpoolmask_model import build_impro_convpoolmask_model
from .convbottle_model import build_impro_convbottle_model


def impro_model_forward_pass(args, impro_model, channels, mask):
    model_name = args.impro_model_name
    if model_name == 'convpool':
        output = impro_model(channels)
    elif model_name == 'convpoolmask':
        output = impro_model(channels, mask)
    elif model_name == 'convbottle':
        output = impro_model(channels)
    else:
        raise ValueError('Model type {} is not supported'.format(model_name))
    # Output of size batch x channel x resolution x resolution
    return output


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def load_impro_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_impro_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    optimiser = build_optim(args, model.parameters())
    optimiser.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimiser


def build_impro_model(args):
    model_name = args.impro_model_name
    if model_name == 'convpool':
        model = build_impro_convpool_model(args)
    elif model_name == 'convpoolmask':
        model = build_impro_convpoolmask_model(args)
    elif model_name == 'convbottle':
        model = build_impro_convbottle_model(args)
    else:
        raise ValueError("Impro model name {} is not a valid option.".format(model_name))
    return model


def build_optim(args, params):
    optimiser = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    # optimiser = torch.optim.SGD(params, args.lr, momentum=.9)
    # optimiser = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)
    return optimiser


