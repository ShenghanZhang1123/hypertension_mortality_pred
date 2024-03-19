from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import os


def roc_curve_plot(y, pred_prob, save_path='./roc_curve.png'):
    fpr, tpr, _ = roc_curve(y, pred_prob)
    roc_auc = auc(fpr, tpr)

    if os.path.isfile(save_path):
        os.remove(save_path)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.00])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def roc_curves_plot(y, pred_prob, y_f, pred_prob_f, save_path='./roc_curve.png'):
    fpr, tpr, _ = roc_curve(y, pred_prob)
    roc_auc = auc(fpr, tpr)

    fpr_f, tpr_f, _ = roc_curve(y_f, pred_prob_f)
    roc_auc_f = auc(fpr_f, tpr_f)

    if os.path.isfile(save_path):
        os.remove(save_path)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f) on all features' % roc_auc)
    plt.plot(fpr_f, tpr_f, lw=2, label='ROC curve (area = %0.2f) on filtered features' % roc_auc_f)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.00])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def precision_recall_curves_plot(y, pred_prob, y_f, pred_prob_f, save_path='./pre_rec_curve.png'):
    if os.path.isfile(save_path):
        os.remove(save_path)
    precision, recall, thresholds = precision_recall_curve(y, pred_prob)
    precision_f, recall_f, thresholds_f = precision_recall_curve(y_f, pred_prob_f)

    # Plot the Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.plot(recall_f, precision_f, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend()
    plt.show()