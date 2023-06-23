import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve, f1_score
import numpy as np

def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))
        # print("pred size",pred.size())
        print('args dataset', args.dataset)
        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')
        elif args.dataset == 'ccd':
            gt = np.load('list/gt-ccd.npy')
        elif args.dataset == 'carla':
            gt = np.load('list/gt-carla.npy')
        else:
            gt = np.load('list/gt-ucf.npy')
        print('gt size', len(list(gt)))
        # exit()

        pred = list(pred.cpu().detach().numpy())
        if args.dataset == 'ccd':
            pred = np.repeat(np.array(pred), 10)
        elif args.dataset == 'carla':
            pred = np.repeat(np.array(pred), 10)
        else:
            pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)
        return rec_auc

def test2(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)
        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            out1, score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            # print(feat_magnitudes)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            # print(logits)
            pred = torch.cat((pred, sig))
            # exit()
        print("pred size",pred.size())
        print('input size', input.size())

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')
        elif args.dataset =='ccd':
            gt = np.load('list/gt-ccd.npy')
        else:
            gt = np.load('list/gt-ucf.npy')
        print('gt size', len(list(gt)))
        print(gt[30300:30400])
        # exit()

        pred = list(pred.cpu().detach().numpy())
        if args.dataset == 'ccd':
            pred = np.repeat(np.array(pred), 10)
        else:
            pred = np.repeat(np.array(pred), 16)
        print(pred[30300:30400])
        ppp = pred[40::50]
        for i, e in enumerate(ppp):
            if e<0.4:
                print(i+1, 2)
        print(pred[734*50:734*50+50])
        print(list(gt)[734*50:734*50+50])
        with open('result.txt', 'w') as f:
            f.write(str(list(gt)[734*50:734*50+50])+'\n')            
            f.write(str(pred[734*50:734*50+50]))            
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save('fpr_best.npy', fpr)
        np.save('tpr_best.npy', tpr)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)
        return rec_auc