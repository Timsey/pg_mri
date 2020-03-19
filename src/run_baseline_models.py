import logging
import time
import datetime
import random
import argparse
import pathlib
import wandb
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from src.helpers.torch_metrics import ssim

from src.helpers.utils import (add_mask_params, save_json, check_args_consistency)
from src.helpers.data_loading import create_data_loaders
from src.recon_models.recon_model_utils import (acquire_new_zf_exp_batch, acquire_new_zf_batch,
                                                recon_model_forward_pass, load_recon_model)
from src.helpers.fastmri_data import DataTransform, SliceData
from src.helpers import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

outputs = defaultdict(lambda: defaultdict(list))

# State used in other experiments when dev data files are shuffled: necessary to compare experimental results when
# sampling rate = 0.04.
STATE = (3, (2855956593, 2685291548, 3345876759, 2476269427, 236192081, 1104449523, 2959661011, 1443336009, 1217332317, 1447679981, 1510771203, 1090512338, 1276969081, 2493916770, 562738784, 1320360969, 767511720, 2134514320, 927086037, 4093062333, 155694031, 2248262844, 3602039969, 3852111709, 747873240, 4147408356, 77841010, 3919474534, 1443286237, 2112741510, 438474856, 4046988663, 1825878883, 643410928, 4257531795, 1730835272, 1953496822, 3371889501, 4236180423, 629250578, 1821410507, 3658734526, 237756976, 1482753473, 4242128630, 1631707468, 864075932, 3898745470, 396198285, 928497869, 919436349, 1052382967, 3308177145, 3638049540, 3111354594, 2389775542, 710343455, 1451877428, 1967980938, 4098556723, 1749783466, 1450662156, 3311927787, 2486387256, 3008679988, 72616703, 1506621182, 3566934370, 2110365442, 460584726, 414300466, 2314312251, 3127334097, 2727877138, 1543123020, 1129883204, 826860018, 3603206241, 988545782, 3574898682, 2406112174, 1692277815, 2326796932, 3019605386, 1696899757, 743649693, 4014550675, 1350688969, 2864172020, 479073098, 2599338912, 21955280, 2884844139, 1643949431, 3069007081, 2817416208, 3742790967, 3787522967, 1598870422, 794309786, 3490636613, 154908901, 3424354041, 4177758159, 1654285064, 101539091, 2951387928, 602633065, 3998001109, 1983396154, 3324009607, 3369758995, 2116372048, 643417482, 1197386399, 1612269876, 3947033543, 3647396235, 4061178602, 3979251351, 255851049, 4200181916, 2557558671, 3678769876, 362774134, 659367709, 1629597157, 3516234936, 46222271, 1043352417, 2763614531, 369883257, 2703291919, 1151286133, 2381074015, 2767568408, 3609152857, 1183679641, 2964830815, 1863537088, 4097676655, 2378891064, 1554640139, 7270970, 3036373658, 2617768134, 1213419942, 1890553713, 1543412096, 141917406, 3382277432, 3659655314, 2978079989, 2887526392, 1666098529, 1993170884, 1192413368, 1059107825, 2791476506, 4265942934, 1124825911, 913721825, 3281640520, 3747125656, 3309843723, 2409455086, 1610404010, 1984453689, 3772326511, 4294766668, 1849186290, 2652101132, 1538756798, 578610553, 3329335359, 45922485, 1195363164, 3513237332, 3154967592, 2898370701, 977381567, 3495957627, 2548389955, 878014215, 1965213043, 1456019979, 1083238756, 2149468347, 2659758064, 1483027423, 3731441276, 2286646308, 333459312, 3916759026, 948811038, 2576830009, 1127096321, 3928834212, 388628886, 102940927, 464866662, 2357359094, 952310908, 693889813, 780004587, 1387327675, 1283664250, 3546322423, 745244265, 4179255440, 3004211711, 3125741961, 1676031225, 2615867736, 3954305252, 1689709596, 139827833, 2577843891, 536599273, 413014686, 4145702992, 3882963794, 65668726, 3487029598, 890186963, 3166468223, 3294958603, 1063422948, 510840780, 569266692, 2753161057, 3311241265, 1251738679, 2843531699, 2266971906, 308804957, 2855931142, 489512739, 3381205381, 3333891042, 1288256681, 3727660094, 3341274118, 2472219463, 1839145297, 3084127340, 2586057489, 368228866, 2527066955, 1459201286, 2318660626, 961685238, 873477391, 1644600890, 771037, 483482729, 21121329, 2919609221, 834852071, 1322665429, 3346091543, 4278420076, 2441204052, 3693209336, 1327210792, 1085171435, 2002147117, 3971326511, 23991618, 3647500598, 236880221, 2302923369, 1493921430, 617042778, 583624870, 3460098418, 3474896004, 2029491413, 826565219, 555941197, 3346835197, 328968225, 3813301698, 3901313081, 2696893000, 1751285250, 1285057645, 655824319, 3864323598, 2341135392, 1847067471, 885264346, 4138367546, 961957916, 3386063195, 1232756346, 3839978031, 1363997935, 2036470010, 1560426674, 1133472663, 1578602295, 3923858488, 1345536850, 1046906566, 1215145175, 2345865021, 3305123764, 1503055309, 2811916830, 4199650218, 817525483, 1312169464, 1810278984, 771141915, 2403942270, 3473392212, 2237795182, 1470256683, 348913771, 2017403863, 1083237564, 603941869, 4261654069, 2711140715, 110237733, 3509925199, 4217511032, 2386139812, 3957873168, 1089203216, 2440324618, 958372223, 736075789, 1467140389, 3164456127, 2937522701, 3561066161, 642579515, 2774877709, 786628895, 1035343136, 2545384106, 1991680536, 341666086, 2728058999, 3807260786, 1870253238, 442608233, 1907015937, 3008327141, 1270063394, 845233525, 1103639348, 4167039377, 456239398, 1071287424, 2750085913, 740793543, 1259758989, 106591988, 4270379569, 2203193507, 1749230411, 2268213419, 2177321872, 2857061901, 3419297000, 3699166438, 3355275273, 76055698, 1819187473, 2648393085, 4180940976, 3141225537, 3024763772, 86830391, 2890524686, 633601087, 1267541716, 538352002, 2953849756, 1884669072, 2560958573, 173351783, 3498170308, 1055035085, 833989337, 2718902338, 778847761, 263222687, 3470547723, 1064738767, 194836893, 346189693, 3904660887, 533113218, 2038448584, 1286187879, 4283929332, 773472458, 3993873262, 1275437075, 3127823695, 34063330, 1671959455, 1037923810, 832669307, 777077728, 2322061253, 3961921395, 1692963438, 2945165646, 3134942549, 747813292, 3093587640, 49397955, 3857420613, 3666761196, 4053726240, 3444851428, 1647754635, 601536823, 591311681, 2370741825, 3015458532, 4155227231, 3225673726, 2179846361, 3591596255, 2196966678, 2380299394, 1400759687, 3787050103, 4047301974, 129217328, 1043565297, 3691431076, 3923043189, 1323903622, 835602098, 3807397497, 1713376201, 1443294726, 957955859, 3259065109, 1717031027, 2695998889, 1056959479, 4043411170, 1561547506, 3649853905, 666159097, 2417127240, 2442012347, 4193009839, 251020519, 4227590991, 2372546743, 1221292960, 2751099824, 3504133702, 1953648909, 908538146, 1813227929, 3037849702, 3960822351, 3660012577, 2459234157, 4293242713, 4206373828, 1710158874, 787125424, 4200525541, 3383342125, 1419054833, 994532323, 4107861776, 3525484381, 2361048268, 441543684, 586238247, 4010991519, 173644219, 2855399821, 1914953730, 3658064521, 1354047878, 1062246725, 2607453393, 3844334008, 3971023544, 1030837882, 1259136130, 1789086425, 1648464946, 1982656668, 173950333, 2496606702, 95389584, 1395565123, 114279173, 907833207, 2324183686, 783575327, 2684165116, 4000824250, 120943132, 1147683754, 3672794051, 4094477426, 1985120019, 3851386714, 1104691727, 1332212981, 3391387447, 695826835, 2165851468, 3780015483, 122808497, 632342701, 351394957, 3713981376, 2520685291, 3623151603, 694235598, 3294283952, 2065778788, 3514559126, 3103381579, 3751003286, 2518384020, 2499107972, 1992535323, 3962909900, 2755430482, 2361416913, 2668981133, 782249255, 3615653525, 1002799289, 1107905989, 1106093211, 2686048015, 2865472966, 2556090126, 1397470054, 1517821371, 4027148997, 3549539035, 1608067839, 426731489, 1369917891, 3683086776, 4221933029, 436978365, 1527991058, 3507110371, 755152754, 1981209586, 266768325, 2921598386, 2523415385, 3932039670, 3858407133, 2315842610, 2993587041, 1211558704, 3596248214, 2732118596, 127614424, 2998810359, 2924137621, 1183009028, 1773744436, 1065789981, 1523294633, 1797830901, 3463127876, 3057459525, 3265870971, 156195655, 1551039731, 1430989965, 3587901389, 4171133276, 1256113261, 171729217, 3455418460, 695902807, 2374661090, 304242212, 874289737, 1509456062, 4156832789, 2516214059, 3295171393, 3055022767, 2228444616, 2511180158, 1806025274, 1448248985, 799958774, 597173038, 3103196783, 3979988283, 2295513662, 1852624270, 2060889614, 2824435803, 2175962721, 1761304134, 2252269084, 682076585, 847154149, 850446753, 3132014216, 4179600318, 1070507432, 2754915747, 1028034752, 3134381292, 295506872, 2480087397, 516), None)


def get_target(args, kspace, masked_kspace, mask, unnorm_gt, gt_mean, gt_std, recon_model, recon, data_range):
    unnorm_recon = recon[:, 0:1, ...] * gt_std + gt_mean  # Back to original scale for metric

    # shape = batch
    base_score = ssim(unnorm_recon, unnorm_gt, size_average=False,
                      data_range=data_range).mean(-1).mean(-1)  # keep channel dim = 1

    res = mask.size(-2)
    batch_acquired_rows = (mask.squeeze(1).squeeze(1).squeeze(-1) == 1)
    acquired_num = batch_acquired_rows[0, :].sum().item()
    tk = res - acquired_num
    batch_train_rows = torch.zeros((mask.size(0), tk)).long().to(args.device)
    for sl, sl_mask in enumerate(mask.squeeze(1).squeeze(1).squeeze(-1)):
        batch_train_rows[sl] = (sl_mask == 0).nonzero().flatten()

    # Acquire chosen rows, and compute the improvement target for each (batched)
    # shape = batch x rows = k x res x res
    zf_exp, _, _ = acquire_new_zf_exp_batch(kspace, masked_kspace, batch_train_rows)
    # shape = batch . tk x 1 x res x res, so that we can run the forward model for all rows in the batch
    zf_input = zf_exp.view(mask.size(0) * tk, 1, res, res)
    # shape = batch . tk x 2 x res x res
    recons_output = recon_model_forward_pass(args, recon_model, zf_input)
    # shape = batch . tk x 1 x res x res, extract reconstruction to compute target
    recons = recons_output[:, 0:1, ...]
    # shape = batch x tk x res x res
    recons = recons.view(mask.size(0), tk, res, res)
    unnorm_recons = recons * gt_std + gt_mean  # TODO: Normalisation necessary?
    gt_exp = unnorm_gt.expand(-1, tk, -1, -1)
    # scores = batch x tk (channels), base_score = batch x 1
    scores = ssim(unnorm_recons, gt_exp, size_average=False, data_range=data_range).mean(-1).mean(-1)
    impros = scores - base_score
    # target = batch x rows, batch_train_rows and impros = batch x tk
    target = torch.zeros((mask.size(0), res)).to(args.device)
    for j, train_rows in enumerate(batch_train_rows):
        # impros[j, 0] (slice j, row 0 in train_rows[j]) corresponds to the row train_rows[j, 0] = 9
        # (for instance). This means the improvement 9th row in the kspace ordering is element 0 in impros.
        kspace_row_inds, permuted_inds = train_rows.sort()
        target[j, kspace_row_inds] = impros[j, permuted_inds]
    return target


def acquire_row(kspace, masked_kspace, next_rows, mask, recon_model):
    zf, mean, std = acquire_new_zf_batch(kspace, masked_kspace, next_rows)
    # Don't forget to change mask for impro_model (necessary if impro model uses mask)
    # Also need to change masked kspace for recon model (getting correct next-step zf)
    # TODO: maybe do this in the acquire_new_zf_batch() function. Doesn't fit with other functions of same
    #  description, but this one is particularly used for this acquisition loop.
    for sl, next_row in enumerate(next_rows):
        mask[sl, :, :, next_row, :] = 1.
        masked_kspace[sl, :, :, next_row, :] = kspace[sl, :, :, next_row, :]
    # Get new reconstruction for batch
    impro_input = recon_model_forward_pass(args, recon_model, zf)  # TODO: args is global here!
    return impro_input, zf, mean, std, mask, masked_kspace


class StepMaskFunc:
    def __init__(self, step, rows, accelerations):
        assert len(rows) == step, 'Mismatch between step and number of acquired rows'
        self.step = step
        self.rows = rows
        assert len(accelerations) == 1, "StepMaskFunc only works for a single acceleration at a time"
        self.acceleration = accelerations[0]
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')
        num_cols = shape[-2]

        # Create the mask
        num_low_freqs = num_cols // self.acceleration
        mask = np.zeros(num_cols)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True
        mask[self.rows] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask


def create_dev_loader(args, step, rows):
    dev_path = args.data_path / f'{args.challenge}_val'
    dev_mask = StepMaskFunc(step, rows, args.accelerations)
    mult = 2 if args.sample_rate == 0.04 else 1  # TODO: this is now hardcoded to get more validation samples: fix this
    dev_sample_rate = args.sample_rate * mult
    dev_data = SliceData(
        root=dev_path,
        transform=DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True),
        sample_rate=dev_sample_rate,
        challenge=args.challenge,
        acquisition=args.acquisition,
        center_volume=args.center_volume,
        state=STATE
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return dev_loader


def run_average_oracle(args, recon_model):
    epoch_outputs = defaultdict(list)
    start = time.perf_counter()

    rows = []
    ssims = np.array([0. for _ in range(args.acquisition_steps + 1)])
    with torch.no_grad():
        for step in range(args.acquisition_steps + 1):
            # Loader for this step: includes starting rows and best rows from previous steps in mask
            loader = create_dev_loader(args, step, rows)
            sum_impros = 0.
            tbs = 0.
            # Find average best improvement over dataset for this step
            for it, data in enumerate(loader):
                # logging.info('Batch {}/{}'.format(it + 1, len(dev_loader)))
                kspace, masked_kspace, mask, zf, gt, _, _, fname, slices = data
                tbs += mask.size(0)
                # shape after unsqueeze = batch x channel x columns x rows x complex
                kspace = kspace.unsqueeze(1).to(args.device)
                masked_kspace = masked_kspace.unsqueeze(1).to(args.device)
                mask = mask.unsqueeze(1).to(args.device)
                # shape after unsqueeze = batch x channel x columns x rows
                zf = zf.unsqueeze(1).to(args.device)
                gt = gt.unsqueeze(1).to(args.device)
                gt, gt_mean, gt_std = transforms.normalize(gt, dims=(-1, -2), eps=1e-11)
                gt = gt.clamp(-6, 6)
                unnorm_gt = gt * gt_std + gt_mean
                data_range = unnorm_gt.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

                # Base reconstruction model forward pass
                recon = recon_model_forward_pass(args, recon_model, zf)

                unnorm_recon = recon[:, 0:1, :, :] * gt_std + gt_mean
                ssim_val = ssim(unnorm_recon, unnorm_gt, size_average=False,
                                data_range=data_range).mean(dim=(-1, -2)).sum()
                ssims[step] += ssim_val.item()

                if step != args.acquisition_steps:  # still acquire, otherwise just need final value, no acquisition
                    output = get_target(args, kspace, masked_kspace, mask, unnorm_gt, gt_mean, gt_std, recon_model,
                                        recon, data_range)
                    output = output.to('cpu').numpy()
                    sum_impros += output.sum(axis=0)
                    epoch_outputs[step + 1].append(output)

            if step != args.acquisition_steps:  # still acquire, otherwise just need final value, no acquisition
                rows.append(np.argmax(sum_impros / tbs))
                print(tbs)

    ssims /= tbs

    for step in range(args.acquisition_steps):
        outputs[-1][step + 1] = np.concatenate(epoch_outputs[step + 1], axis=0).tolist()
    save_json(args.run_dir / 'preds_per_step_per_epoch.json', outputs)

    return ssims, time.perf_counter() - start


def run_oracle(args, recon_model, dev_loader):
    """
    Evaluates using SSIM of reconstruction over trajectory. Doesn't require computing targets!
    """

    ssims = 0
    epoch_outputs = defaultdict(list)
    start = time.perf_counter()
    tbs = 0
    with torch.no_grad():
        for it, data in enumerate(dev_loader):
            # logging.info('Batch {}/{}'.format(it + 1, len(dev_loader)))
            kspace, masked_kspace, mask, zf, gt, _, _, fname, slices = data
            tbs += mask.size(0)
            # shape after unsqueeze = batch x channel x columns x rows x complex
            kspace = kspace.unsqueeze(1).to(args.device)
            masked_kspace = masked_kspace.unsqueeze(1).to(args.device)
            mask = mask.unsqueeze(1).to(args.device)
            # shape after unsqueeze = batch x channel x columns x rows
            zf = zf.unsqueeze(1).to(args.device)
            gt = gt.unsqueeze(1).to(args.device)
            gt, gt_mean, gt_std = transforms.normalize(gt, dims=(-1, -2), eps=1e-11)
            gt = gt.clamp(-6, 6)
            unnorm_gt = gt * gt_std + gt_mean
            data_range = unnorm_gt.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

            # Base reconstruction model forward pass
            recon = recon_model_forward_pass(args, recon_model, zf)

            unnorm_recon = recon[:, 0:1, :, :] * gt_std + gt_mean
            init_ssim_val = ssim(unnorm_recon, unnorm_gt, size_average=False,
                                 data_range=data_range).mean(dim=(-1, -2)).sum()

            batch_ssims = [init_ssim_val.item()]

            for step in range(args.acquisition_steps):
                if args.model_type == 'oracle':
                    output = get_target(args, kspace, masked_kspace, mask, unnorm_gt, gt_mean, gt_std, recon_model,
                                        recon, data_range)
                elif args.model_type == 'center_sym':  # Always get high score for most center unacquired row
                    acquired = mask[0].squeeze().nonzero().flatten()
                    output = torch.tensor([[(mask.size(-2) - 1) / 2 - abs(i - 0.1 - (mask.size(-2) - 1) / 2)
                                            for i in range(mask.size(-2))]
                                           for _ in range(mask.size(0))])
                    output[:, acquired] = 0.
                    output = output.to(args.device)
                elif args.model_type == 'center_asym_left':
                    acquired = mask[0].squeeze().nonzero().flatten()
                    output = torch.tensor([[i if i <= mask.size(-2) // 2 else 0
                                            for i in range(mask.size(-2))]
                                           for _ in range(mask.size(0))])
                    output[:, acquired] = 0.
                    output = output.to(args.device)
                elif args.model_type == 'center_asym_right':
                    acquired = mask[0].squeeze().nonzero().flatten()
                    output = torch.tensor([[mask.size(-2) - i if i >= mask.size(-2) // 2 else 0
                                            for i in range(mask.size(-2))]
                                           for _ in range(mask.size(0))])
                    output[:, acquired] = 0.
                    output = output.to(args.device)
                elif args.model_type == 'random':  # Generate random scores (set acquired to 0. to perform filtering)
                    acquired = mask[0].squeeze().nonzero().flatten()
                    output = torch.randn((mask.size(0), mask.size(-2)))
                    output[:, acquired] = 0.
                    output = output.to(args.device)

                epoch_outputs[step + 1].append(output.to('cpu').numpy())
                # Greedy policy (size = batch)
                _, next_rows = torch.max(output, dim=1)

                # Acquire this row
                impro_input, zf, _, _, mask, masked_kspace = acquire_row(kspace, masked_kspace, next_rows, mask,
                                                                         recon_model)
                unnorm_recon = impro_input[:, 0:1, :, :] * gt_std + gt_mean
                # shape = 1
                ssim_val = ssim(unnorm_recon, unnorm_gt, size_average=False,
                                data_range=data_range).mean(dim=(-1, -2)).sum()
                # eventually shape = al_steps
                batch_ssims.append(ssim_val.item())

            # shape = al_steps
            ssims += np.array(batch_ssims)

    ssims /= tbs

    for step in range(args.acquisition_steps):
        outputs[-1][step + 1] = np.concatenate(epoch_outputs[step + 1], axis=0).tolist()
    save_json(args.run_dir / 'preds_per_step_per_epoch.json', outputs)

    return ssims, time.perf_counter() - start


def main(args):
    # Reconstruction model
    recon_args, recon_model = load_recon_model(args)
    check_args_consistency(args, recon_args)

    # Add mask parameters for training
    args = add_mask_params(args, recon_args)

    # Create directory to store results in
    savestr = 'res{}_al{}_accel{}_{}_{}_{}'.format(args.resolution, args.acquisition_steps, args.accelerations,
                                                   args.model_type, args.recon_model_name,
                                                   datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    args.run_dir = args.exp_dir / savestr
    args.run_dir.mkdir(parents=True, exist_ok=False)

    if args.wandb:
        wandb.config.update(args)

    # Logging
    logging.info(args)
    logging.info(recon_model)
    logging.info('Model type: {}'.format(args.model_type))

    # Save arguments for bookkeeping
    args_dict = {key: str(value) for key, value in args.__dict__.items()
                 if not key.startswith('__') and not callable(key)}
    save_json(args.run_dir / 'args.json', args_dict)

    # Initialise summary writer
    writer = SummaryWriter(log_dir=args.run_dir / 'summary')

    if args.model_type == 'oracle_average':
        oracle_dev_ssims, oracle_time = run_average_oracle(args, recon_model)
    else:
        # Create data loader
        _, dev_loader, _, _ = create_data_loaders(args)
        oracle_dev_ssims, oracle_time = run_oracle(args, recon_model, dev_loader)

    dev_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(oracle_dev_ssims)])
    logging.info(f'  DevSSIM = [{dev_ssims_str}]')
    logging.info(f'DevSSIMTime = {oracle_time:.2f}s')

    # For storing in wandb
    for epoch in range(args.num_epochs + 1):
        logging.info(f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] ')
        if args.model_type == 'random':
            oracle_dev_ssims, oracle_time = run_oracle(args, recon_model, dev_loader)
            dev_ssims_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(oracle_dev_ssims)])
            logging.info(f'DevSSIMTime = {oracle_time:.2f}s')
        logging.info(f'  DevSSIM = [{dev_ssims_str}]')
        if args.wandb:
            wandb.log({'val_ssims': {str(key): val for key, val in enumerate(oracle_dev_ssims)}}, step=epoch + 1)
    writer.close()


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images')
    parser.add_argument('--dataset', choices=['fastmri', 'cifar10'], required=True,
                        help='Dataset to use.')
    parser.add_argument('--wandb',  action='store_true',
                        help='Whether to use wandb logging for this run.')

    parser.add_argument('--model-type', choices=['center_sym', 'center_asym_left', 'center_asym_right',
                                                 'random', 'oracle', 'oracle_average'], required=True,
                        help='Type of model to use.')

    # Data parameters
    parser.add_argument('--challenge', type=str, default='singlecoil',
                        help='Which challenge for fastMRI training.')
    parser.add_argument('--data-path', type=pathlib.Path, default=None,
                        help='Path to the dataset. Required for fastMRI training.')
    parser.add_argument('--sample-rate', type=float, default=1.,
                        help='Fraction of total volumes to include')
    parser.add_argument('--acquisition', type=str, default='CORPD_FBK',
                        help='Use only volumes acquired using the provided acquisition method. Options are: '
                             'CORPD_FBK, CORPDFS_FBK (fat-suppressed), and not provided (both used).')

    # Reconstruction model
    parser.add_argument('--recon-model-checkpoint', type=pathlib.Path, default=None,
                        help='Path to a pretrained reconstruction model. If None then recon-model-name should be'
                        'set to zero_filled.')
    parser.add_argument('--recon-model-name', choices=['kengal_laplace', 'kengal_gauss', 'zero_filled'], required=True,
                        help='Reconstruction model name corresponding to model checkpoint.')
    parser.add_argument('--in-chans', default=1, type=int, help='Number of image input channels')

    parser.add_argument('--center-volume', action='store_true',
                        help='If set, only the center slices of a volume will be included in the dataset. This '
                             'removes the most noisy images from the data.')
    parser.add_argument('--use-recon-mask-params', action='store_true',
                        help='Whether to use mask parameter settings (acceleration and center fraction) that the '
                        'reconstruction model was trained on. This will overwrite any other mask settings.')

    # Mask parameters, preferably they match the parameters the reconstruction model was trained on. Also see
    # argument use-recon-mask-params above.
    parser.add_argument('--accelerations', nargs='+', default=[8], type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for '
                             'each volume.')
    parser.add_argument('--reciprocals-in-center', nargs='+', default=[1], type=float,
                        help='Inverse fraction of rows (after subsampling) that should be in the center. E.g. if half '
                             'of the sampled rows should be in the center, this should be set to 2. All combinations '
                             'of acceleration and reciprocals-in-center will be used during training (every epoch a '
                             'volume randomly gets assigned an acceleration and center fraction.')

    parser.add_argument('--acquisition-steps', default=10, type=int, help='Acquisition steps to train for per image.')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')

    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers to use for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, required=True,
                        help='Directory where model and results should be saved. Will create a timestamped folder '
                        'in provided directory each run')

    parser.add_argument('--verbose', type=int, default=1,
                        help='Set verbosity level. Lowest=0, highest=4."')
    return parser


if __name__ == '__main__':
    # To fix known issue with h5py + multiprocessing
    # See: https://discuss.pytorch.org/t/incorrect-data-using-h5py-with-dataloader/7079/2?u=ptrblck
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn')

    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    if args.wandb:
        wandb.init(project='mrimpro', config=args)

    # To get reproducible behaviour, additionally set args.num_workers = 0 and disable cudnn
    # torch.backends.cudnn.enabled = False
    main(args)
