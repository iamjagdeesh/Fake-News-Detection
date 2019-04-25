import numpy as np
import pandas as pd


class AdjacencyMatrix:
    def __init__(self, base_path):
        self.base_path = base_path

    def get_folder_data(self, folder):
        news_df = pd.read_csv(self.base_path + folder + "/News.txt", header=None)
        news_list = list(news_df[0])

        users_df = pd.read_csv(self.base_path + folder + "/User.txt", header=None)
        users_list = list(users_df[0])

        news_user_df = pd.read_csv(self.base_path + folder + "/" + folder + "NewsUser.txt", header=None, sep="\t")
        result_list = []
        for index, row in news_user_df.iterrows():
            news_index = row[0]
            user_index = row[1]
            result_list.append([news_list[news_index - 1], users_list[user_index - 1]])

        result_df = pd.DataFrame(result_list, columns=["News", "Users"])

        return result_df, news_df, users_df

    def get_all_data(self):
        # print("Fetching BuzzFeed data")
        # bf_res_df, bf_news_df, bf_users_df = self.get_folder_data("BuzzFeed")

        print("Fetching PolitiFact data")
        pf_res_df, pf_news_df, pf_users_df = self.get_folder_data("PolitiFact")
        #
        # news_user_df = pd.concat([bf_res_df, pf_res_df])
        # news_df = pd.concat([bf_news_df, pf_news_df])
        # users_df = pd.concat([bf_users_df, pf_users_df])
        #
        # return news_user_df, news_df, users_df

        return pf_res_df, pf_news_df, pf_users_df

    # def get_adjacency_matrix(self):
    #     news_user_df, news_df, users_df = self.get_all_data()
    #     news_list = list(news_df[0].unique())
    #     users_list = list(users_df[0].unique())
    #
    #     nodes = news_list + users_list
    #     result = np.empty((len(nodes), len(nodes)))
    #
    #     print("Generating Adjacency Matrix")
    #     for index, row in news_user_df.iterrows():
    #         news = row[0]
    #         user = row[1]
    #         result[nodes.index(news)][nodes.index(user)] = 1
    #         result[nodes.index(user)][nodes.index(news)] = 1
    #
    #     result_df = pd.DataFrame(result, columns=nodes, index=nodes)
    #
    #     print("Dumping the Adjacency Matrix to CSV")
    #     result_df.to_csv(self.base_path + "adjacency_matrix.csv")
    #
    #     print("Done")
    #
    #     return result_df

    def get_connected_news(self, news, df):
        res = set()
        user_for_news_df = df[df["News"] == news]
        users = user_for_news_df["Users"].unique()

        for user in users:
            user_df = df[df["Users"] == user]
            for item in user_df["News"].unique():
                res.add(item)

        return list(res)

    def get_adjacency_matrix(self):
        news_user_df, news_df, users_df = self.get_all_data()
        news_list = list(news_df[0].unique())

        result = np.empty((len(news_list), len(news_list)))

        print("Generating Adjacency Matrix")
        for news in news_list:
            connected_news = self.get_connected_news(news, news_user_df)
            for connected_news in connected_news:
                result[news_list.index(news)][news_list.index(connected_news)] = 1
                result[news_list.index(connected_news)][news_list.index(news)] = 1

        result_df = pd.DataFrame(result, columns=news_list, index=news_list)

        print("Dumping the Adjacency Matrix to CSV")
        result_df.to_csv(self.base_path + "news_news_pf_adjacency_matrix.csv")

        print("Done")

        return result_df


if __name__ == "__main__":
    base_path = "../dataset/"

    adj = AdjacencyMatrix(base_path)
    res = adj.get_adjacency_matrix()