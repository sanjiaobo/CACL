import torch
import numpy as np
import math

def entropy_compute_class(outputs,labels,i,epoch):
    with torch.no_grad():
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



        alpha_t = 0.5 * (1 - (i-1) / epoch)
        #alpha_t = 0.5
        pred = torch.softmax(outputs,dim=1)
        entropys = -torch.sum(pred * torch.log(pred + 1e-10), dim=1)

        entropy_total = []
        for iii in range(len(label_total)):
            en = []
            for jjj in a[iii]:
                en.append(entropys[jjj])
            entropy_total.append(en)

        # for m in range(len(entropys)):
        #     entropys[m] = entropys[m]*proportion[int(labels[m])]
        for i in range(len(label_total)):
            if len(label_total[i]) != 0:
                thresh_total.append(np.percentile(entropys[label_total[i]].cpu().numpy().flatten(), 100 * (1 - alpha_t)))
        reliable_total = []
        un_reliable_total = []
        for y in range(len(label_total)):
            reliable = []
            un_reliable = []
            for i in label_total[y]:

                if entropys[i] <= thresh_total[y]:
                    reliable.append(i)
                else:
                    un_reliable.append(i)
            reliable_total.append(reliable)
            un_reliable_total.append(un_reliable)
        reliable_label = []
        for aa in range(len(reliable_total)):
            for a in reliable_total[aa]:
                reliable_label.append(labels[a])
        if len(reliable_label) == 0:
            print("entropy_total",entropy_total,"thresh_total",thresh_total,"reliable_total",reliable_total,"un_reliable_total",un_reliable_total)
        un_reliable_label = []
        for bb in range(len(un_reliable_total)):
            for c in un_reliable_total[bb]:
                un_reliable_label.append(labels[c])
        if len(un_reliable_label) == 0:
            print("entropy_total",entropy_total,"thresh_total",thresh_total,"reliable_total",reliable_total,"un_reliable_total",un_reliable_total)
        # if len(un_reliable_total) == 0:
        #     print("熵是",entropys,"阈值是",thresh_total,"类别",label_total)
        # mid_feature_total = []
        # for y in range(len(reliable_total)):
        #     mid_feature = torch.zeros(1, 512).cuda()
        #     for i in reliable_total[y]:
        #         mid_feature += feature[i]
        #     mid_feature_total.append(mid_feature/len(reliable_total[y]))

        return reliable_total,un_reliable_total,entropys


def entropy_compute(outputs,labels,i,epoch):
    with torch.no_grad():
        alpha_t = 0.5 * (1 - (i-1) / epoch)
        pred = torch.softmax(outputs,dim=1)
        entropys = -torch.sum(pred * torch.log(pred + 1e-10), dim=1)

        # for m in range(len(entropys)):
        #     entropys[m] = entropys[m]*proportion[int(labels[m])]
        thresh = np.percentile(entropys.cpu().numpy().flatten(),100*(1-alpha_t))
        reliable = []
        un_reliable = []
        for i in range(len(entropys)):
            if entropys[i] < thresh:  #不能加等于，不然可能出现没有unreliable的情况
                reliable.append(i)
            else:
                un_reliable.append(i)

        return entropys,thresh,reliable,un_reliable,pred

def update_centers(centers):

    SB = 0
    # l2 = tf.reduce_mean(tf.reduce_sum(tf.square(self.centers-mu),axis=1)
    for i in range(len(centers)):
        for j in range(i, len(centers)):
            if i != j:
                dist = torch.sum(torch.square(centers[i] - centers[j]))  # scalar
                SB = SB +torch.exp(-dist / 512)

    return SB/21



