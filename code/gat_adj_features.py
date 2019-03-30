import pandas as pd
import numpy as np
import scipy.sparse as sp
from code.adjacency_matrix import AdjacencyMatrix
from code.feature_matrix import FeatureMatrix

class GATInputGenerator:
    def __init__(self):
        self.gat = "cora"
        self.AM = AdjacencyMatrix(base_path = "/Users/jagde/Documents/ASU/SWM/Project/Fake-News-Detection/dataset/")
        self.FM = FeatureMatrix(base_path = "/Users/jagde/Documents/ASU/SWM/Project/Fake-News-Detection/dataset/")
        self.label_zip = None

    def getAdj(self):
        adj_df = self.AM.get_adjacency_matrix()
        # feature_df = self.FM.get_feature_matrix()
        print(adj_df.shape)
        adj_np = adj_df.values
        print(type(adj_np))
        res = []
        for i in range(adj_np.shape[0]):
            #print("Adj ",i," Row ", adj_np[i])
            temp_res = np.nonzero(np.array(adj_np[i]))[0]
            #print("temp", temp_res)
            # res = np.concatenate((res, temp_res), axis=0)
            res.append(list(temp_res))
        print(res)

        return sp.csr_matrix(res)

    def getFeatures(self):
        feature_df = self.FM.get_feature_matrix()
        # print(adj_df.shape)
        label = feature_df['label'].tolist()
        label_comp = [0 if each else 1 for each in label]
        self.label_zip = list(zip(label_comp, label))
        feature_df.drop(['label'], axis=1)
        feature_np = feature_df.values
        # print(type(feature_np))
        res = feature_np
        for i in range(feature_np.shape[0]):
            #print("Adj ",i," Row ", adj_np[i])
            temp_res = np.nonzero(np.array(feature_np[i]))[0]
            #print("temp", temp_res)
            # res = np.concatenate((res, temp_res), axis=0)
            res.append(list(temp_res))
        print(res)

        return sp.csr_matrix(res)

    def getYs(self):
        yTrain = yVal = yTest = self.label_zip[:]
        train_mask, val_mask, test_mask = [False] * len(yTrain)
        n = len(yTrain)
        train_range = range(0, int(n * 0.5))
        val_range = range(int(n * 0.5), int(n * 0.75))
        test_range = range(int(n * 0.75), n)

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

    def getComps(self):
        adj = self.getAdj()
        features = self.getFeatures()
        yTrain, yVal, yTest, train_mask, val_mask, test_mask = self.getYs()

        return adj, features, yTrain, yVal, yTest, train_mask, val_mask, test_mask

if __name__ == "__main__":
    obj = GATInputGenerator()
    obj.getComps()