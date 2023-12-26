import torch
import torch.nn.functional as F
import numpy as np


class Noisy_loss(object):
    def __call__(self, outputs_noisy, outputs_aug):
        # output_x: tensor
        # softlabel_x: tensor , size like output_x
        noisy = torch.softmax(outputs_noisy, dim=1)
        #print("噪声图片",noisy)
        aug = torch.softmax(outputs_aug, dim = 1)
        #print("特征增强",aug)
        noisy_loss = torch.mean((noisy-aug)**2)
        return noisy_loss

class Distribution_loss(object):
    def __call__(self, feature,labels, mid_feature,un_reliable,reliable,epoch,entropy):
        # label_0 = []
        # label_1 = []
        # label_2 = []
        # label_3 = []
        # label_4 = []
        # label_5 = []
        # label_6 = []
        # a = []
        # label_total = []
        # thresh_total = []
        # for h in range(len(labels)):
        #     if labels[h] == 0:
        #         label_0.append(h)
        #     elif labels[h] == 1:
        #         label_1.append(h)
        #     elif labels[h] == 2:
        #         label_2.append(h)
        #     elif labels[h] == 3:
        #         label_3.append(h)
        #     elif labels[h] == 4:
        #         label_4.append(h)
        #     elif labels[h] == 5:
        #         label_5.append(h)
        #     else:
        #         label_6.append(h)
        #
        # a.append(label_0)
        # a.append(label_1)
        # a.append(label_2)
        # a.append(label_3)
        # a.append(label_4)
        # a.append(label_5)
        # a.append(label_6)
        #
        # for ii in range(len(a)):
        #     if len(a[ii]) != 0:
        #         label_total.append(a[ii])
        #
        # distribution_loss = 0
        # coef = feature.exp().mean(dim=1,keepdim=True) #越小代表越可靠
        # coef_1 = torch.exp(-coef)
        # loss = 0
        # for i in range(len(label_total)):
        #     for aa in label_total[i]:
        #       loss += (torch.exp(-coef[aa])) * torch.norm((feature[aa] - mid_feature[i])) ** 2


        coef = feature.var(dim=1, keepdim=True)  # 越小代表越可靠
        le = 0
        loss_unreliable = 0
        loss_reliable = 0
        for i in range(len(un_reliable)):
            if len(un_reliable[i]) > 0:
                for a in un_reliable[i]:
                    loss_unreliable += (torch.exp(-coef[a])) * torch.norm((feature[a] - mid_feature[i])) ** 2
                    # loss = torch.norm((feature[a] - mid_feature[i]))**2
                le = le + len(un_reliable[i])

        for i in range(len(reliable)):
            if len(reliable[i]) > 0:
                for j in reliable[i]:
                    loss_reliable += torch.norm((feature[j]-mid_feature[i]))**2



        #noisy_loss = loss_unreliable / le + loss_reliable/(len(feature)-le)
        loss_un = loss_unreliable / le
        loss_re = loss_reliable/(len(feature)-le)
        #noisy_loss = loss / len(labels)
        noisy_loss = loss_unreliable / le

        return noisy_loss

class Distribution_center_loss(object):
    def __call__(self, feature,labels, mid_feature,un_reliable,reliable,epoch,entropy):
        label_0 = []
        label_1 = []
        label_2 = []
        label_3 = []
        label_4 = []
        label_5 = []
        label_6 = []
        a = []
        label_total = []
        thresh_total = []
        for h in range(len(labels)):
            if labels[h] == 0:
                label_0.append(h)
            elif labels[h] == 1:
                label_1.append(h)
            elif labels[h] == 2:
                label_2.append(h)
            elif labels[h] == 3:
                label_3.append(h)
            elif labels[h] == 4:
                label_4.append(h)
            elif labels[h] == 5:
                label_5.append(h)
            else:
                label_6.append(h)

        a.append(label_0)
        a.append(label_1)
        a.append(label_2)
        a.append(label_3)
        a.append(label_4)
        a.append(label_5)
        a.append(label_6)

        for ii in range(len(a)):
            if len(a[ii]) != 0:
                label_total.append(a[ii])

        loss = 0
        for i in range(len(label_total)):
            for aa in label_total[i]:
              loss += torch.norm((feature[aa] - mid_feature[i])) ** 2


        noisy_loss = loss / len(feature)

        return noisy_loss