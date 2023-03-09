import matplotlib.pyplot as plt
import  numpy as np
from pylab import mpl
from sklearn import svm
class draw_fig_SVM:
    def __int__(self):
        pass


    def Visual_SVM(self, feature, label, models, title = ''):
        mpl.rcParams['font.sans-serif'] = ['SimHei']             # 指定默认字体
        mpl.rcParams['axes.unicode_minus'] = False
        fig, sub = plt.subplots(1, 1,figsize=(12,8))
        X0, X1 = feature[:, 0], feature[:, 1]
        xx, yy = self.make_meshgrid(X0, X1)

        # for clf, title, ax in zip(models, titles, sub):
        clf = models
        ax = sub
        self.plot_contours(sub, clf, xx, yy,alpha=0.8)
        ax.scatter(X0, X1, c=label, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel(u'feature_X')
        ax.set_ylabel(u'feature_Y')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        plt.show()


    def make_meshgrid(self, x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
        return xx, yy


    def plot_contours(self, ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contour(xx, yy, Z, **params)
        return out


class heatmap_figure:
    def __init__(self):
        pass



