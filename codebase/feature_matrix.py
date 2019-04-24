import json
import os

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from bert import run_classifier
from bert import tokenization


class FeatureMatrix:
    def __init__(self, base_path):
        self.base_path = base_path

    def get_data_from_file(self, file):
        with open(file) as json_file:
            data = json.load(json_file)
            if file.split("/")[-2] == "FakeNewsContent":
                return [data['text'], 1]
            else:
                return [data['text'], 0]

    def get_folder_data(self, folder):
        data_list = []
        for subfolder in ["FakeNewsContent", "RealNewsContent"]:
            for file in os.listdir(self.base_path + folder + "/" + subfolder):
                if file.endswith(".json"):
                    data_list.append(self.get_data_from_file(self.base_path + folder + "/" + subfolder + "/" + file))

        data_df = pd.DataFrame(data_list, columns=["text", "label"])

        return data_df

    def get_all_data(self):
        print("Fetching BuzzFeed data")
        bf_data_df = self.get_folder_data("BuzzFeed")

        # print("Fetching PolitiFact data")
        # pf_data_df = self.get_folder_data("PolitiFact")

        # all_data_df = pd.concat([bf_data_df, pf_data_df])

        return bf_data_df

    def create_tokenizer_from_hub_module(self):
        with tf.Graph().as_default():
            bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1")
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                      tokenization_info["do_lower_case"]])

        return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def get_feature_matrix(self, dataset = "BuzzFeed"):
        all_data_df = self.get_folder_data(folder=dataset)
        all_data_df = all_data_df.sample(frac=1)

        inputExamples = all_data_df.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                                text_a=x['text'],
                                                                                text_b=None,
                                                                                label=x['label']),
                                          axis=1)
        tokenizer = self.create_tokenizer_from_hub_module()
        features = run_classifier.convert_examples_to_features(inputExamples, [0, 1], 128, tokenizer)

        train_features_list = []
        for item in features:
            temp = item.input_ids
            temp.append(item.label_id)
            train_features_list.append(temp)
        column_names = ["feature" + str(i) for i in range(128)]
        column_names.append("label")
        features_df = pd.DataFrame(train_features_list, columns=column_names)

        return features_df


if __name__ == "__main__":
    base_path = "../dataset/"

    adj = FeatureMatrix(base_path)
    res = adj.get_feature_matrix("PolitiFact")
