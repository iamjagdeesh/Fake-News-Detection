import scipy.sparse as sp
import pandas as pd
from feature_matrix import FeatureMatrix
import random

class GATInputGenerator:
    def __init__(self):
        self.gat = "cora"
        self.FM = FeatureMatrix(base_path = "../dataset/")
        self.label_zip = None

    def getAdj(self, dataset="BuzzFeed"):
        if dataset == "BuzzFeed":
            adj_np = pd.read_csv("../dataset/news_news_bf_adjacency_matrix.csv", header=None).values
        else:
            adj_np = pd.read_csv("../dataset/news_news_pf_adjacency_matrix.csv", header=None).values

        return sp.csr_matrix(adj_np, dtype=int)

    def getFeatures(self, dataset="BuzzFeed"):
        feature_df = self.FM.get_feature_matrix(dataset)
        label = feature_df['label'].tolist()
        label_comp = [0 if each else 1 for each in label]
        self.label_zip = list(zip(label_comp, label))
        feature_df.drop(['label'], axis=1)
        feature_np = feature_df.values

        return sp.csr_matrix(feature_np, dtype=float).tolil()


    def getYs(self, dataset="BuzzFeed"):
        if dataset == "BuzzFeed":
            random.seed(1)
        else:
            random.seed(1)
        yTrain = self.label_zip[:]
        yVal = self.label_zip[:]
        yTest = self.label_zip[:]
        train_mask = [False] * len(yTrain)
        val_mask = [False] * len(yTrain)
        test_mask = [False] * len(yTrain)
        n = len(yTrain)

        set_of_records_range = set(range(n))

        train_range = set(random.sample(set_of_records_range, k=int(n * 0.6)))
        set_of_records_range = set_of_records_range - train_range

        val_range = set(random.sample(set_of_records_range, k=int(n * 0.2)))
        set_of_records_range = set_of_records_range - train_range

        test_range = set(random.sample(set_of_records_range, k=int(n * 0.2)))

        for i in train_range:
            yVal[i] = (0,0)
            yTest[i] = (0,0)
            train_mask[i] = True
        for i in val_range:
            yTrain[i] = (0,0)
            yTest[i] = (0,0)
            val_mask[i] = True
        for i in test_range:
            yVal[i] = (0,0)
            yTrain[i] = (0,0)
            test_mask[i] = True

        return yTrain, yVal, yTest, train_mask, val_mask, test_mask

    def getComps(self, dataset="BuzzFeed"):
        print(dataset)
        adj = self.getAdj(dataset)
        features = self.getFeatures(dataset)

        yTrain, yVal, yTest, train_mask, val_mask, test_mask = self.getYs(dataset)

        return adj, features, yTrain, yVal, yTest, train_mask, val_mask, test_mask

if __name__ == "__main__":
    obj = GATInputGenerator()
    obj.getComps(dataset="BuzzFeed")