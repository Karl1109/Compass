import numpy as np
import os
import logging
import glob
import cv2

def cal_global_acc(pred, gt):
    h,w = gt.shape
    return [np.sum(pred==gt), float(h*w)]

def get_statistics_seg(pred, gt, num_cls=2):
    statistics = []
    for i in range(num_cls):
        tp = np.sum((pred==i)&(gt==i))
        fp = np.sum((pred==i)&(gt!=i))
        fn = np.sum((pred!=i)&(gt==i))
        statistics.append([tp, fp, fn])
    return statistics

def get_statistics_prf(pred, gt):
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn]

def segment_metrics(pred_list, gt_list, num_cls = 2):
    global_accuracy_cur = []
    statistics = []

    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_img = (pred / 255).astype('uint8')
        global_accuracy_cur.append(cal_global_acc(pred_img, gt_img))
        statistics.append(get_statistics_seg(pred_img, gt_img, num_cls))

    global_acc = np.sum([v[0] for v in global_accuracy_cur]) / np.sum([v[1] for v in global_accuracy_cur])
    counts = []
    for i in range(num_cls):
        tp = np.sum([v[i][0] for v in statistics])
        fp = np.sum([v[i][1] for v in statistics])
        fn = np.sum([v[i][2] for v in statistics])

        counts.append([tp, fp, fn])

    mean_acc = np.sum([v[0] / (v[0] + v[2]) for v in counts]) / num_cls
    mean_iou_acc = np.sum([v[0] / (np.sum(v)) for v in counts]) / num_cls

    return global_acc, mean_acc, mean_iou_acc

def prf_metrics(pred_list, gt_list):
    statistics = []

    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_img = (((pred / np.max(pred))>0.1)).astype('uint8')
        statistics.append(get_statistics_prf(pred_img, gt_img))

    tp = np.sum([v[0] for v in statistics])
    fp = np.sum([v[1] for v in statistics])
    fn = np.sum([v[2] for v in statistics])
    p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
    r_acc = tp / (tp + fn)
    f_acc = 2 * p_acc * r_acc / (p_acc + r_acc)
    return p_acc,r_acc,f_acc

def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        statistics = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            statistics.append(get_statistics(pred_img, gt_img))
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])

        p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        r_acc = tp / (tp + fn)
        final_accuracy_all.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc)])

    return final_accuracy_all

def thred_half(src_img_list, tgt_img_list):
    Precision, Recall, F_score = prf_metrics(src_img_list, tgt_img_list)
    Global_Accuracy, Class_Average_Accuracy, Mean_IOU = segment_metrics(src_img_list, tgt_img_list)
    print("Global Accuracy:{}, Class Average Accuracy:{}, Mean IOU:{}, Precision:{}, Recall:{}, F score:{}".format(
        Global_Accuracy, Class_Average_Accuracy, Mean_IOU, Precision, Recall, F_score))

def get_statistics(pred, gt):
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn]

def cal_mIoU_metrics(pred_list, gt_list, thresh_step=0.01, pred_imgs_names=None, gt_imgs_names=None):
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []
        for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            TP = np.sum((pred_img == 1) & (gt_img == 1))
            TN = np.sum((pred_img == 0) & (gt_img == 0))
            FP = np.sum((pred_img == 1) & (gt_img == 0))
            FN = np.sum((pred_img == 0) & (gt_img == 1))
            if (FN + FP + TP) <= 0:
                iou = 0
            else:
                iou_1 = TP / (FN + FP + TP)
                iou_0 = TN / (FN + FP + TN)
                iou = (iou_1 + iou_0)/2
            iou_list.append(iou)
        ave_iou = np.mean(np.array(iou_list))
        final_iou.append(ave_iou)
    mIoU = np.max(np.array(final_iou))
    return mIoU

def imread(path, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
    im = cv2.imread(path, load_mode)
    if convert_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if load_size > 0:
        im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    if thresh > 0:
        _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    return im

def get_image_pairs(data_dir, suffix_gt='real_B', suffix_pred='fake_B'):
    gt_list = glob.glob(os.path.join(data_dir, '*{}.png'.format(suffix_gt)))
    pred_list = [ll.replace(suffix_gt, suffix_pred) for ll in gt_list]
    assert len(gt_list) == len(pred_list)
    pred_imgs, gt_imgs = [], []
    pred_imgs_names, gt_imgs_names = [], []
    for pred_path, gt_path in zip(pred_list, gt_list):
        pred_imgs.append(imread(pred_path))
        gt_imgs.append(imread(gt_path, thresh=127))
        pred_imgs_names.append(pred_path)
        gt_imgs_names.append(gt_path)
    return pred_imgs, gt_imgs, pred_imgs_names, gt_imgs_names

def eval(log_eval, results_dir, epoch):
    suffix_gt = "lab"
    suffix_pred = "pre"
    log_eval.info(results_dir)
    log_eval.info("checkpoints _ " + results_dir)
    src_img_list, tgt_img_list, pred_imgs_names, gt_imgs_names = get_image_pairs(results_dir, suffix_gt, suffix_pred)
    assert len(src_img_list) == len(tgt_img_list)

    Precision, Recall, F_score = prf_metrics(src_img_list, tgt_img_list)
    mIoU = cal_mIoU_metrics(src_img_list, tgt_img_list, pred_imgs_names=pred_imgs_names, gt_imgs_names=gt_imgs_names)
    log_eval.info("mIouU _ " + str(mIoU))
    log_eval.info("F1 _ " + str(F_score))
    log_eval.info("eval finish!")

    return {'epoch': epoch, 'mIoU': mIoU, 'F1': F_score}

if __name__ == '__main__':
    suffix_gt = "lab"
    suffix_pred = "pre"
    results_dir = "results"
    logging.info(results_dir)
    src_img_list, tgt_img_list, pred_imgs_names, gt_imgs_names = get_image_pairs(results_dir, suffix_gt, suffix_pred)
    assert len(src_img_list) == len(tgt_img_list)

    Precision, Recall, F_score = prf_metrics(src_img_list, tgt_img_list)
    mIoU = cal_mIoU_metrics(src_img_list, tgt_img_list, pred_imgs_names=pred_imgs_names, gt_imgs_names=gt_imgs_names)
    print("mIouU _ " + str(mIoU))
    print("F1 _ " + str(F_score))
    print("Finish!")
