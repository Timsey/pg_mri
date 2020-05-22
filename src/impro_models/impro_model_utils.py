import torch
import shutil

from .convpool_model import build_impro_convpool_model
from .convpoolmask_model import build_impro_convpoolmask_model
from .convbottle_model import build_impro_convbottle_model
from .mask_model import build_impro_maskfc_model
from .maskconv_model import build_impro_maskconv_model
from .convpoolmaskconv_model import build_impro_convpoolmaskconv_model
from .location_model import build_impro_location_model
from .multimaskconv_model import build_impro_multimaskconv_model
from ..helpers.elioptim import EliOptimizer


def impro_model_forward_pass(args, impro_model, channels, mask):
    model_name = args.impro_model_name
    train = True
    if model_name == 'convpool':
        output = impro_model(channels)
    elif model_name == 'convpoolmask':
        output = impro_model(channels, mask)
    elif model_name == 'convbottle':
        output = impro_model(channels)
    elif model_name == 'maskfc':
        # Channels not actually used
        output = impro_model(channels, mask)
    elif model_name == 'maskconv':
        output = impro_model(channels, mask)
    elif model_name == 'maskconvunit':
        unitmask = torch.ones_like(mask)
        output = impro_model(channels, unitmask)
    elif model_name == 'multimaskconv':
        output = impro_model(channels, mask)
    elif model_name == 'convpoolmaskconv':
        output = impro_model(channels, mask)
    elif model_name == 'location':
        output = impro_model(channels, mask)
    elif model_name == 'center':  # Always get high score for most center unacquired row
        acquired = mask[0].squeeze().nonzero().flatten()
        output = torch.tensor([[(mask.size(1) - 1) / 2 - abs(i - 0.1 - (mask.size(1) - 1) / 2)
                                for i in range(mask.size(1))]
                              for _ in range(mask.size(0))])
        output[:, acquired] = 0.
        output = output.to(args.device)
        train = False
    elif model_name == 'random':  # Generate random scores (set acquired to 0. to perform filtering)
        acquired = mask[0].squeeze().nonzero().flatten()
        output = torch.randn((mask.size(0), mask.size(1)))
        output[:, acquired] = 0.
        output = output.to(args.device)
        train = False
    else:
        raise ValueError('Model type {} is not supported'.format(model_name))
    # Output of size batch x channel x resolution x resolution
    return output, train


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best, milestones):
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

    if epoch in milestones:
        torch.save(
            {
                'epoch': epoch,
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dev_loss': best_dev_loss,
                'exp_dir': exp_dir
            },
            f=exp_dir / f'model_{epoch}.pt'
        )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def load_impro_model(checkpoint_file, optim=False):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_impro_model(args)

    # No gradients for this model
    for param in model.parameters():
        param.requires_grad = False

    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    start_epoch = checkpoint['epoch']

    if optim:
        optimizer = build_optim(args, model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        return model, args, start_epoch, optimizer

    del checkpoint
    return model, args


def build_impro_model(args):
    model_name = args.impro_model_name
    if model_name == 'convpool':
        model = build_impro_convpool_model(args)
    elif model_name == 'convpoolmask':
        model = build_impro_convpoolmask_model(args)
    elif model_name == 'convbottle':
        model = build_impro_convbottle_model(args)
    elif model_name == 'maskfc':
        model = build_impro_maskfc_model(args)
    elif model_name in ['maskconv', 'maskconvunit']:
        model = build_impro_maskconv_model(args)
    elif model_name == 'multimaskconv':
        model = build_impro_multimaskconv_model(args)
    elif model_name == 'convpoolmaskconv':
        model = build_impro_convpoolmaskconv_model(args)
    elif model_name == 'location':
        model = build_impro_location_model(args)
    elif model_name == 'center':
        model = model_name
    elif model_name == 'random':
        model = model_name
    else:
        raise ValueError("Impro model name {} is not a valid option.".format(model_name))
    return model


def build_optim(args, params):
    optimiser = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    # optimiser = EliOptimizer(params, lr=args.lr)
    # optimiser = torch.optim.SGD(params, args.lr, momentum=.9)
    # optimiser = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)
    return optimiser


