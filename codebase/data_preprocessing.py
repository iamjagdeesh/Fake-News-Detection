import pandas as pd
import os
import json
import bert.extract_features as xf
import tokenization
class DataPreProcessing(object):
    def __init__(self):
        self.data_set_loc = "dataset"
        self.fake_news_loc = os.path.join( "..",self.data_set_loc, "BuzzFeed",  "FakeNewsContent")
        self.real_news_loc = os.path.join( "..", self.data_set_loc, "BuzzFeed", "RealNewsContent")
        self.fake_news_desc_list = []
        self.real_news_desc_list = []


    def extract_desc_from_files(self, file_path, data_list):
        with open(file_path) as fd:
            json_data = json.load(fd)
           # data_list.append(xf.convert_examples_to_features(json_data['text'], 1, tokenizer))
            data_list.append(json_data['text'].replace("\n", " "))
    def read_json_news_files(self):
        for file in os.listdir(self.fake_news_loc):
            if file.endswith(".json"):
                self.extract_desc_from_files(os.path.join(self.fake_news_loc, file), self.fake_news_desc_list)
        for file in os.listdir(self.real_news_loc):
            if file.endswith(".json"):
                self.extract_desc_from_files(os.path.join(self.real_news_loc, file), self.real_news_desc_list)
        fake_news_df = pd.DataFrame({0: self.fake_news_desc_list})
        fake_news_df.to_csv("fake_news.csv", index=False, header=False)
        fake_news_df.columns = ["text"]
        fake_news_df["label"] = "Fake"
        real_news_df = pd.DataFrame({0: self.real_news_desc_list})
        real_news_df.columns = ["text"]
        real_news_df["label"] = "Real"
        combined_df = pd.concat([real_news_df, fake_news_df])
        a=1

    def test_bert(self):
        examples = xf.read_examples(os.path.join(self.fake_news_loc,"fake_news.csv"))
        features = xf.convert_examples_to_features(examples, 91, tokenization.Tokenizer)
        a=1

if __name__ == "__main__":
    ob = DataPreProcessing()
    ob.test_bert()