from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model, Model2
from custom_dataset import Dataset
from train_custom import train, train_two
from test_10crop import test, test2
import option
from tqdm import tqdm
from utils import Visualizer
from config import *

viz = Visualizer(env='eval', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)

    # train_snloader = DataLoader(Dataset(args, test_mode=False, is_normal=True, dataset='carla'),
    #                             batch_size=args.batch_size, shuffle=True,
    #                             num_workers=0, pin_memory=False, drop_last=True)
    # train_saloader = DataLoader(Dataset(args, test_mode=False, is_normal=False, dataset='carla'),
    #                            batch_size=args.batch_size, shuffle=True,
    #                            num_workers=0, pin_memory=False, drop_last=True)
    # train_tnloader = DataLoader(Dataset(args, test_mode=False, is_normal=True, dataset='ccd'),
    #                            batch_size=args.batch_size, shuffle=True,
    #                            num_workers=0, pin_memory=False, drop_last=True)
    # train_taloader = DataLoader(Dataset(args, test_mode=False, is_normal=False, dataset='ccd'),
    #                            batch_size=args.batch_size, shuffle=True,
    #                            num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True, dataset='ccd'),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = Model2(args.feature_size, args.batch_size)
    model.load_state_dict(torch.load('ckpt/rrrr240-i3d.pkl'))

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = ''   # put your own path here
    auc = test2(test_loader, model, args, viz, device)
    
    # for step in tqdm(
    #         range(1, len(test_loader) + 1),
    #         total=len(test_loader),
    #         dynamic_ncols=True
    # ):
    #     if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
    #         for param_group in optimizer.param_groups:
    #             param_group["lr"] = config.lr[step - 1]

    #     if (step - 1) % len(train_snloader) == 0:
    #         loadersn_iter = iter(train_snloader)
    #     if (step - 1) % len(train_saloader) == 0:
    #         loadersa_iter = iter(train_saloader)
            
    #     if (step - 1) % len(train_snloader) == 0:
    #         loadertn_iter = iter(train_tnloader)
    #     if (step - 1) % len(train_saloader) == 0:
    #         loaderta_iter = iter(train_taloader)

    #     train_two(loadersn_iter, loadersa_iter, loadertn_iter, loaderta_iter, model, args.batch_size, optimizer, scheduler, viz, device)

    #     # if step % 5 == 0 and step > 200:
    #     if step > 200:

    #         auc = test2(test_loader, model, args, viz, device)
    #         test_info["epoch"].append(step)
    #         test_info["test_AUC"].append(auc)

    #         if test_info["test_AUC"][-1] > best_AUC:
    #             best_AUC = test_info["test_AUC"][-1]
    #             torch.save(model.state_dict(), './ckpt/' + args.model_name + '{}-i3d.pkl'.format(step))
    #             save_best_record(test_info, os.path.join(output_path, '{}-step-AUC.txt'.format(step)))
    # torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')

