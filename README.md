# Fake-News-Detection

## CSE 573: Semantic Web Mining Project

## Group 10
1. Abhijith Shreesh (ASU ID: 1213204276)
2. Aditya Chayapathy (ASU ID: 1213050538)
3. Anuhya Sai (ASU ID: 1212931887)
4. Arun Karthick Manickam Alagar Muthumanickam (ASU ID: 1213135077)
5. Jagdeesh Basavaraju (ASU ID: 1213004713)

## Description
The project aims at classifying the given news articles as fake or true based on the content and users associated with it using Graph Attention Networks (GATs).

1. Extracted the content of news articles from the given dataset.
2. Vectorized the news article content using BERT to obtain feature vector for every article.
3. Derived relationship among news articles based on the users the articles are associated with.
4. Classified the news articles by feeding the feature vectors and relationship matrix to the GAT.
5. Compared and contrasted the performance of GAT against traditional machine learning algorithms.

Technology used: Google BERT, Graph Attention Network (GAT), Python, Pandas, NumPy, scikit-learn, Tensorflow

## Steps to execute
1. Go to the folder named "codebase".
2. Run the command "pip install -r requirements.txt && python execute_bf_pf.py BuzzFeed".
3. The above command will install all the requirements and run GAT on Buzzfeed dataset.
4. Run the command "python execute_bf_pf.py PolitiFact".
5. The above command will run GAT on PolitiFact dataset.
6. After running the above commands on each dataset, results on training, validation and test set will be displayed.
