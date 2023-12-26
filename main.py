import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import time
import warnings

from torch.utils.tensorboard import SummaryWriter
from resnet18_7_plot.utils import *
from resnet18_7_plot.loss import *
from resnet18_7_plot.entropy_compute import entropy_compute_class, update_centers
from resnet18_7_plot.noisy_loss import Distribution_loss,Distribution_center_loss
from ramp import ramp_up, ramp_down
#from resnet18_1.draw_loss_acc import draw_loss
import os
import json
from resnet18_7_plot.utils1 import draw_matrix
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from thop import profile

# seed = int(8)
# torch.manual_seed(seed) # 为CPU设置随机种子
# torch.cuda.manual_seed(seed)
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Facial Expression Recognition Experiment')
parser.add_argument('--Log_Name', type=str, default='EM18_RAF_run', help='Naming Format: date_experiment_model')
parser.add_argument('--Experiment', default='EM', type=str, choices=['EM', 'AU', 'Fuse'],
                    help='1->Expression Recognition Experiment, 2->AU Recognition Experiment, 3->Feature Fuse Experiment')
parser.add_argument('--Dataset', default='RAF', type=str,
                    choices=['RAF', 'SFEW', 'MMI', 'ExpW', 'BP4D', 'AffectNet', 'FERPlus'],
                    help='Value Range: RAF, BP4D, SFEW, MMI, ExpW')
parser.add_argument('--Distribute', default='Basic', type=str, choices=['Basic', 'Compound'],
                    help='Value Range: Basic, Compound')
parser.add_argument('--Aligned', default=True, type=str2bool, help='whether to Aligned Image')
parser.add_argument('--Model', default='ResNet-18', type=str, choices=['ResNet-101', 'ResNet-50', 'ResNet-18'],
                    help='1->ResNet-101(pre-trained on ImageNet), 2->ResNet-50(pre-trained on ImageNet), 3->ResNet-18(pre-trained on ImageNet)')
parser.add_argument('--Resume_Model', default='ms1m_res18.pkl', type=str,
                    help='if Resume_Model == none, then load pre-trained on ImageNet from PyTorch')
parser.add_argument('--Dim', default=512, type=int, help='Dim Of Fuse Feature')
parser.add_argument('--numOfAU', default=17, type=int, help='Number of Action Units')
parser.add_argument('--numOfLabel', default=7, type=int, help='Number of Expression Labels')
parser.add_argument('--Epoch', default=60, type=int, help='Epoch')
parser.add_argument('--LearnRate', default=0.1, type=float, help='Learning Rate')
parser.add_argument('--Train_Batch_Size', default=64, type=int, help='Batch Size during training')
parser.add_argument('--Test_Batch_Size', default=64, type=int, help='Batch Size during testing')
parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--Num_Workers', default=0, type=int, help='Number of Workers')
parser.add_argument('--DataParallel', default=False, type=str2bool, help='Data Parallel')


# TODO: 定义 t-sne 函数

def tsne_plot(outputs, targets, epoch, args, acc):
    tsne = TSNE()
    tsne_output = tsne.fit_transform(outputs)
    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets
    ax = sns.scatterplot(
        x='x', y='y',
        hue='targets',
        # TODO: 8 类别数
        palette=sns.color_palette("hls", 7),
        data=df,
        marker='o',
        legend="full",
        alpha=0.9,
        s=5
    )
    la = ax.legend()
    # TODO: 类别名
    la.get_texts()[0].set_text('Su')
    la.get_texts()[1].set_text('Fe')
    la.get_texts()[2].set_text('Di')
    la.get_texts()[3].set_text('Ha')
    la.get_texts()[4].set_text('Sa')
    la.get_texts()[5].set_text('An')
    la.get_texts()[6].set_text('Ne')
    #la.get_texts()[7].set_text('7')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('epochs: {}'.format(epoch))
    if args.Dataset == "RAF":
        plt.savefig('/home/lab501/photo/{}-{}-tsne.png'.format(epoch, acc), dpi=1000, bbox_inches='tight')
        plt.close()

    if args.Dataset == "AffectNet":
        plt.savefig('/home/lab501/photo_res18/{}-{}-7tsne.png'.format(epoch, acc), dpi=1000, bbox_inches='tight')
        plt.close()
    plt.show()


def Train(args, model, criterion, optimizer, train_loader, epoch, criter, train_acc, train_loss):
    distribution_loss = Distribution_center_loss()
    numOfClass = args.numOfAU if args.Experiment == 'AU' else args.numOfLabel  # 7
    acc_1, acc_2, prec, recall = [AverageMeter() for i in range(numOfClass)], [AverageMeter() for i in
                                                                               range(numOfClass)], [AverageMeter() for i
                                                                                                    in range(
            numOfClass)], [AverageMeter() for i in range(numOfClass)]
    loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()
    ramp_up_w, ramp_down_w = ramp_up(epoch, 20), ramp_down(epoch, 20)
    # ramp_up_w, ramp_down_w = 1, 1
    model.train()
    # if args.Experiment == 'Fuse':
    #     model.backbone.eval()
    for i in range(numOfClass):
        acc_1[i].reset()
        acc_2[i].reset()
        prec[i].reset()
        recall[i].reset()
    loss.reset()
    data_time.reset()
    batch_time.reset()
    optimizer, lr = Adjust_Learning_Rate(optimizer, epoch, args.LearnRate)
    end = time.time()
    for step, (input, au_loc, target) in tqdm(enumerate(train_loader, start=1)):
        input, imgPath = input
        input, target = input.cuda(), target.cuda()
        data_time.update(time.time() - end)
        if args.Experiment in ['AU', 'Fuse']:
            # au_loc = au_loc.cuda()
            au_target = model.get_au_target(target.cpu())  # generate au label
        # forward
        if args.Experiment == 'EM':
            feature, pred = model(input)
            # pred = model(input)
            reliable, un_reliable, entropy = entropy_compute_class(pred, target, epoch, args.Epoch)
            _,preds = torch.max(pred,dim=1)

            mid_feature_total = []
            for y in range(len(reliable)):
                mid_feature = torch.zeros(1, 512).cuda()
                for ii in reliable[y]:
                    mid_feature += feature[ii]
                mid_feature_total.append(mid_feature / len(reliable[y]))
            # mid_feature_total = []
            # for y in range(len(reliable)):
            #     mid_feature = torch.zeros(1, 512).cuda()
            #     for ii in reliable[y]:
            #         mid_feature += feature[ii]
            #     if len(un_reliable[y]) > 0:
            #         for iii in un_reliable[y]:
            #             mid_feature += feature[iii]
            #         mid_feature_total.append(mid_feature/(len(reliable[y])+len(un_reliable[y])))
            #     else:
            #         mid_feature_total.append(mid_feature/len(reliable[y]))


            center_loss = 0.05 * update_centers(mid_feature_total)
            distribution = 0.005 * distribution_loss(feature, target, mid_feature_total, un_reliable, reliable,epoch, entropy)
            loss_ = criterion(pred,target)#+ distribution + center_loss  # +criter(au, au_target)+0.5 * Expression_Independent_AU_Loss()(au, au_target)
            # loss_hard = criterion(un_reliable_outputs, un_reliable_label)# + 0.0001 * distribution_loss(feature,target,mid_feature_total,un_reliable,reliable,epoch,entropy)
            # loss_easy = criterion(reliable_outputs, reliable_label) #+ a
            # loss_ = ramp_down_w*loss_easy + ramp_up_w*loss_hard
            # loss_ = loss_easy + loss_hard

        # backward
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        # compute accuracy, recall and loss
        if args.Experiment == 'EM':
            Compute_Accuracy_Expression(args, pred, target, loss_, acc_1, acc_2, prec, recall, loss)
        batch_time.update(time.time() - end)
        end = time.time()
    Accuracy_Info, acc_1_avg, acc_2_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc_1, acc_2, prec, recall,
                                                                                      numOfClass=numOfClass)
    # writer

    # writer.add_scalar('Accuracy_1', acc_1_avg, epoch)
    # writer.add_scalar('Accuracy_2', acc_2_avg, epoch)
    # writer.add_scalar('Precision', prec_avg, epoch)
    # writer.add_scalar('Recall', recall_avg, epoch)
    # writer.add_scalar('F1', f1_avg, epoch)
    # writer.add_scalar('Loss', loss.avg, epoch)

    LogInfo = '''
    [Tain ({exp})]: 
    Epoch {0}
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})
    Learning Rate {1}\n'''.format(epoch, lr, data_time=data_time, batch_time=batch_time, exp=args.Experiment)

    LogInfo += Accuracy_Info
    LogInfo += '''    Acc_avg(1) {0:.4f} Acc_avg(2) {1:.4f} Prec_avg {2:.4f} Recall_avg {3:.4f} F1_avg {4:.4f}
    Loss {loss.avg:.4f}'''.format(acc_1_avg, acc_2_avg, prec_avg, recall_avg, f1_avg, loss=loss)
    train_acc.append(acc_1_avg)
    train_loss.append(loss)
    return acc_1_avg, acc_2_avg, prec_avg, recall_avg, f1_avg, loss.avg, lr
    #print(LogInfo)


def Test(args, model, criterion, optimizer, test_loader, epoch, Best_Accuracy, criter, test_acc, test_loss):
    numOfClass = args.numOfAU if args.Experiment == 'AU' else args.numOfLabel
    acc_1, acc_2, prec, recall = [AverageMeter() for i in range(numOfClass)], [AverageMeter() for i in
                                                                               range(numOfClass)], [AverageMeter() for i
                                                                                                    in range(
            numOfClass)], [AverageMeter() for i in range(numOfClass)]
    loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''
    混淆矩阵
    '''
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels_json = [label for _, label in class_indict.items()]
    NHX_M = [[0 for i in range(7)] for j in range(7)]

    # Test Model
    model.eval()
    for i in range(numOfClass):
        acc_1[i].reset()
        acc_2[i].reset()
        prec[i].reset()
        recall[i].reset()
    loss.reset()
    data_time.reset()
    batch_time.reset()
    end = time.time()

    # TODO: 1024 换成 features 的长度
    # 用来保存一个epochs里的所有outputs和targets
    outputs = torch.tensor([[0 for _ in range(512)]]).to(device)
    targets = torch.tensor([]).to(device)

    for step, (input, au_loc, target) in enumerate(test_loader, start=1):
        input, imgPath = input
        input, target = input.cuda(), target.cuda()
        data_time.update(time.time() - end)
        if args.Experiment in ['AU', 'Fuse']:
            au_loc = au_loc.cuda()
            au_target = model.get_au_target(target.cpu())  # generate au label
        with torch.no_grad():
            # forward
            if args.Experiment == 'EM':
                features, pred = model(input)  # 7维

                # TODO: 保存每一个batch的数据
                # outputs是二维数组，targets一维数组
                outputs = torch.cat((outputs, features), axis=0)
                targets = torch.cat((targets, target.to(device)))

                ''''''  ##############获得类别
                pred_classes = torch.max(pred, dim=1)[1]
                for i in range(len(target)):  # 求的是混淆矩阵的整数形式
                    NHX_M[target[i]][pred_classes[i]] += 1
                # pred = model(input)
                loss_ = criterion(pred,target)  # + criter(au, au_target) + 0.5 * Expression_Independent_AU_Loss()(au, au_target)

        # compute accuracy, recall and loss
        if args.Experiment == 'EM':
            Compute_Accuracy_Expression(args, pred, target, loss_, acc_1, acc_2, prec, recall, loss)

        batch_time.update(time.time() - end)
        end = time.time()
    Accuracy_Info, acc_1_avg, acc_2_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc_1, acc_2, prec, recall,
                                                                                      numOfClass=numOfClass)
    '''
    混淆矩阵的概率
    '''
    for stepi, i in enumerate(NHX_M):
        sumi = sum(i)
        for stepj, ii in enumerate(i):
            p = ii / sumi
            NHX_M[stepi][stepj] = round(p*100, 2)  # 保留两位小数
    # writer
    '''
    writer.add_scalar('Accuracy_1', acc_1_avg, epoch)
    writer.add_scalar('Accuracy_2', acc_2_avg, epoch)
    writer.add_scalar('Precision', prec_avg, epoch)
    writer.add_scalar('Recall', recall_avg, epoch)
    writer.add_scalar('F1', f1_avg, epoch)
    writer.add_scalar('Loss', loss.avg, epoch)
    '''
    LogInfo = '''
    [Test ({exp})]: 
    Epoch {0}
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})\n'''.format(epoch, data_time=data_time,
                                                                       batch_time=batch_time, exp=args.Experiment)

    LogInfo += Accuracy_Info
    LogInfo += '''    Acc_avg(1) {0:.4f} Acc_avg(2) {1:.4f} Prec_avg {2:.4f} Recall_avg {3:.4f} F1_avg {4:.4f}
    Loss {loss.avg:.4f}'''.format(acc_1_avg, acc_2_avg, prec_avg, recall_avg, f1_avg, loss=loss)
    test_acc.append(acc_1_avg)
    test_loss.append(loss)
    #print(LogInfo)

    # TODO: 数据格式变换
    outputs = outputs[1:].cpu().numpy()
    targets = targets.cpu().numpy()

    # Save Checkpoints
    if acc_1_avg > Best_Accuracy:
        if args.Dataset == "RAF" and acc_1_avg > 0.88:
            # TODO: 调用t t-sne 绘制
            tsne_plot(outputs, targets, epoch, args, acc_1_avg)
            draw_matrix(NHX_M, 7, labels_json, args, epoch, acc_1_avg)
        if args.Dataset == "AffectNet":
            # TODO: 调用t t-sne 绘制
            tsne_plot(outputs, targets, epoch, args, acc_1_avg)
            draw_matrix(NHX_M, 7, labels_json, args, epoch, acc_1_avg)
        Best_Accuracy, Best_Epoch = acc_1_avg, epoch
        print('[Save] Best Acc: %.4f, Best Epoch: %d' % (Best_Accuracy, Best_Epoch))
        # if isinstance(model, nn.DataParallel):
        #     torch.save(model.module.state_dict(), 'trainmodel/{}.pkl'.format(args.Log_Name))
        # else:
        #     torch.save(model.state_dict(), 'trainmodel/{}.pkl'.format(args.Log_Name))

    # #TODO: 数据格式变换
    # outputs = outputs[1:].cpu().numpy()
    # targets = targets.cpu().numpy()

    # TODO: 返回 outputs 和 targets
    return Best_Accuracy, acc_1_avg, acc_2_avg, prec_avg, recall_avg, f1_avg, loss.avg, NHX_M  # , outputs, targets


def main():
    '''main'''
    # Parse Argument
    args = parser.parse_args()
    distribution_loss = Distribution_loss()
    # Experiment Information
    print('Log Name: %s' % args.Log_Name)
    print('Backbone: %s' % args.Model)
    print('Experiment: %s' % args.Experiment)
    print('Resume_Model: %s' % args.Resume_Model)
    print('CUDA_VISIBLE_DEVICES: %s' % args.GPU_ID)
    print('================================================')
    print('Dataset: %s' % args.Dataset)
    print('Distribute: %s' % args.Distribute)
    print('Use Aligned Image' if args.Aligned else 'Don\'t use Aligned Image')
    print('================================================')
    if args.Distribute == 'Basic':
        args.numOfLabel = 7
    elif args.Distribute == 'Compound':
        args.numOfLabel = 11
    print('Dim: %d' % args.Dim)
    print('Number Of Action Units: %d' % args.numOfAU)
    print('Number Of Expression Labels: %d' % args.numOfLabel)
    print('================================================')
    print('Number of Workers: %d' % args.Num_Workers)
    print('Use Data Parallel' if args.DataParallel else 'Dont\'t use Data Parallel')
    print('Epoch: %d' % args.Epoch)
    print('Train Batch Size: %d' % args.Train_Batch_Size)
    print('Test Batch Size: %d' % args.Test_Batch_Size)
    print('================================================')
    # Bulid Model
    print('Load Model...')
    model = Bulid_Model(args)
    # print(model)
    print('Done!')
    print('================================================')

    # Set Optimizer
    print('Building Optimizer...')
    param_optim = Set_Param_Optim(args, model)
    optimizer = Set_Criterion_Optimizer(args, param_optim)
    print('Done!')
    print('================================================')
    # Bulid Dataloader
    print("Buliding Train and Test Dataloader...")
    if args.Dataset == 'ExpW':
        train_loader, test_loader = BulidDataloader(args)
    else:
        train_loader = BulidDataloader(args, flag='train')
        test_loader = BulidDataloader(args, flag='test')
    print('Done!')
    print('================================================')
    Best_Accuracy = 0

    # if args.Experiment in ['EM', 'Fuse']:
    criterion = nn.CrossEntropyLoss()
    # elif args.Experiment == 'AU':
    #    criterion = MSELoss()

    criter = MSELoss()
    # Running Experiment
    print("Run Experiment...")
    writer = SummaryWriter('tensorboard/{}'.format(args.Log_Name))
    '''
    设置tensorboard
    '''
    train_writer = SummaryWriter(log_dir="res_train/train")
    val_writer = SummaryWriter(log_dir="res_val/val")
    '''
    混淆矩阵
    '''
    # json_label_path = './class_indices.json'
    # assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    # json_file = open(json_label_path, 'r')
    # class_indict = json.load(json_file)
    # labels_json = [label for _, label in class_indict.items()]

    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    for epoch in range(1, args.Epoch + 1):
        acc_1_avg, acc_2_avg, prec_avg, recall_avg, f1_avg, loss_train, train_lr = Train(args, model, criterion,
                                                                                         optimizer, train_loader, epoch,
                                                                                         criter, train_acc, train_loss)
        train_writer.add_scalar('Accuracy_1', acc_1_avg, epoch)
        # train_writer.add_scalar('Accuracy_2', acc_2_avg, epoch)
        # train_writer.add_scalar('Precision', prec_avg, epoch)
        # train_writer.add_scalar('Recall', recall_avg, epoch)
        # train_writer.add_scalar('F1', f1_avg, epoch)
        train_writer.add_scalar('Loss', loss_train, epoch)
        # train_writer.add_scalar('lr', train_lr, epoch)
        # NM为混淆矩阵
        Best_Accuracy, val_acc1, val_acc2, val_prec, val_recall, val_f1, loss_val, NM = Test(args, model, criterion,
                                                                                             optimizer, test_loader,
                                                                                             epoch, Best_Accuracy,
                                                                                             criter, test_acc,
                                                                                             test_loss)

        # # TODO: 调用t t-sne 绘制
        # tsne_plot(outputs, targets, epoch)

        val_writer.add_scalar('Accuracy_1', val_acc1, epoch)
        # val_writer.add_scalar('Accuracy_2', val_acc2, epoch)
        # val_writer.add_scalar('Precision', val_prec, epoch)
        # val_writer.add_scalar('Recall', val_recall, epoch)
        # val_writer.add_scalar('F1', val_f1, epoch)
        val_writer.add_scalar('Loss', loss_val, epoch)
        torch.cuda.empty_cache()
    # draw_loss(train_acc,train_loss,test_acc,test_loss,args.Epoch)
    # plot(NM, num_classes=8, labels=labels_json)


if __name__ == '__main__':
    for i in range(30):
        main()