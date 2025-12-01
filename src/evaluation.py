
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
#@title Performance Evaluation

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, accuracy_score, f1_score, classification_report, log_loss
import random

class ClassificationEvaluator:
    def __init__(self, classifiername, y_actual_label, y_pred_scores=None, y_pred_label=None):
        self.name = classifiername
        self.y_test = y_actual_label
        self.y_scores = y_pred_scores
        self.y_pred = y_pred_label
        self.y_test_np10,self.y_scores_np10 = self.random_neg_pos_ratio()
        plt.rcParams.update({'font.size': 22})

    def plot_confusion_matrix(self):
        if self.y_pred is None:
            raise ValueError("To compute confusion_matrix, provide predicted classes (y_pred).")

        cm = confusion_matrix(self.y_test, self.y_pred)
        title='Normalized Confusion Matrix '+ self.name,
        cmap = plt.cm.Oranges
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize = (8, 18))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, size = 24)
        plt.colorbar(aspect=4)
        classes = ['Neg','Pos']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0, size = 17)
        plt.yticks(tick_marks, classes, size = 17)
        fmt = '.2f' #if normalize else 'd'
        thresh = cm.max() / 2.

        # Labeling the plot
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), fontsize = 24,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.grid(None); plt.tight_layout()
        plt.ylabel('Actual Label', size = 22)
        plt.xlabel('Predicted Label', size = 21)
        plt.show()

    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_scores)
        roc_auc = roc_auc_score(self.y_test, self.y_scores)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='%s' % self.name + ' (AUROC: %0.3f)' % roc_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(prop = { "size": 16 }, loc ="lower right"); plt.tight_layout()
        plt.show()

    def plot_prc_curve(self):
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_scores)
        aupr = average_precision_score(self.y_test, self.y_scores)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='%s' % self.name + ' (AUPR: %0.3f)' % aupr)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(prop = { "size": 16 }, loc ="lower left"); plt.tight_layout()
        plt.show()

    def random_neg_pos_ratio(self):
        N_ratio = 10
        indices_of_pos = np.where(self.y_test == 1)[0]
        indices_of_neg = np.where(self.y_test == 0)[0]
        if len(indices_of_neg) > N_ratio*len(indices_of_pos):
            indices_of_10neg = random.sample(list(indices_of_neg),N_ratio*len(indices_of_pos))
        else:
            indices_of_10neg = indices_of_neg
        print("Size of raw Neg:Pos ",len(indices_of_neg),len(indices_of_pos))
        combined_indices = np.array(list(indices_of_pos) + list(indices_of_10neg))
        print("Size of ratio_N Neg:Pos ",len(indices_of_10neg),len(indices_of_pos))
        return self.y_test[combined_indices],self.y_scores[combined_indices]
    
    def closest_idx(self,lst, K):#print()
        lst = np.asarray(lst)
        idx = (np.abs(lst - K)).argmin()
        return idx
    
    def plot_roc_curve_ratio(self):
        fpr, tpr, threshold1 = roc_curve(self.y_test_np10, self.y_scores_np10)
        roc_auc = roc_auc_score(self.y_test_np10, self.y_scores_np10)

        for i in [0.00001,0.0001,0.0005,0.001,0.005,0.05,0.01]:
            idx = self.closest_idx(fpr,i);
            print("FPR/TPR/score at FPR of interest: %s, %.6f,%.6f, %.0f" % (i,fpr[idx],tpr[idx],threshold1[idx])) 
        for s in [1,5,10,20,30,50,100,300]:
            idx = self.closest_idx(threshold1,s);
            print("FPR/TPR/score at score of interest: %s, %.6f,%.6f, %.1f" % (s,fpr[idx],tpr[idx],threshold1[idx])) 

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='%s' % self.name + ' (AUROC: %0.3f)' % roc_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(prop = { "size": 16 }, loc ="lower right"); plt.tight_layout()
        plt.show()

    def plot_prc_curve_ratio(self):
        precision, recall, _ = precision_recall_curve(self.y_test_np10, self.y_scores_np10)
        aupr = average_precision_score(self.y_test_np10, self.y_scores_np10)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='%s' % self.name + ' (AUPR: %0.3f)' % aupr)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(prop = { "size": 16 }, loc ="lower left"); plt.tight_layout()
        plt.show()

    def calculate_accuracy(self):
        if self.y_pred is None:
            raise ValueError("To compute accuracy_score, provide predicted classes (y_pred).")
        else:
            ac=accuracy_score(self.y_test, self.y_pred)
            print('accuracy_score:',ac)
            return ac

    def calculate_f1_score(self):
        if self.y_pred is None:
            raise ValueError("To compute f1_score, provide predicted classes (y_pred).")
        else:
            f1=f1_score(self.y_test, self.y_pred)
            print('f1_score:',f1)
            return f1

    def generate_classification_report(self):
        if self.y_pred is None:
            raise ValueError("To generate classification report, provide predicted classes (y_pred).")
        else:
            report = classification_report(self.y_test, self.y_pred)
            print('classification_report:',report)
            return report

    def compute_mse_r2(self):
        if self.y_scores is None:
            raise ValueError("To compute mean squred error and r2, provide predicted classes (y_pred).")
        else:
            mse = mean_squared_error(self.y_test, self.y_scores)
            r2 = r2_score(self.y_test, self.y_scores)
            print('mean_squared_error:',mse)
            print('R-squared (R2):',r2)
            return classification_report(self.y_test, self.y_scores)
    def compute_cross_entropy(self):
        # Calculate the cross-entropy
        cross_entropy = log_loss(self.y_test, self.y_scores)
        print(f"Cross-Entropy of prediction: {cross_entropy}")

    def plot_pred_distributions(self):
        list_zip = zip(self.y_test,self.y_scores)
        df_pred = pd.DataFrame(list_zip, columns = ['Label', 'LR'])
        df_pred_0 = df_pred[df_pred['Label']==0]['LR']
        df_pred_1 = df_pred[df_pred['Label']==1]['LR']

        # Create two separate histograms with left and right y-axes
        fig, ax1 = plt.subplots(figsize=(8, 6))
        # Plot histogram for Class A on the left y-axis
        ax1.hist(df_pred_0, bins='doane', alpha=0.9, color='royalblue', label='Neg')
        ax1.set_xlabel('Prediction Score')
        ax1.set_ylabel('Counts (for Neg)', color='royalblue')
        ax1.tick_params(axis='y', labelcolor='royalblue')
        ax1.set_yscale('log')

        # Create a second y-axis on the right
        ax2 = ax1.twinx()
        ax2.hist(df_pred_1, bins='doane', alpha=0.5, color='green', label='Pos')
        ax2.set_ylabel('Counts (for Pos)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_yscale('log')
        plt.title('Distributions of Predictions')
        fig.tight_layout()
        plt.show()

