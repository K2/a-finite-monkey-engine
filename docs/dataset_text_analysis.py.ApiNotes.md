This script performs lexical analysis on the `.text` sections from dataset entries where `veryGood` is True and `quality` is +1 or higher.
It executes the following steps:
1. Loads the dataset from `/reports/dataset.jsonl` using the jsonlines module.
2. Filters entries to include only those meeting the criteria.
3. Uses TF-IDF vectorization to represent the text.
4. Clusters the texts with KMeans to bucket similar entries.
5. Extracts the top terms in each cluster to serve as labels for significant property descriptions.
6. Prints a table grouping entries by cluster, along with cluster labels and a brief summary of each entry.

This tool facilitates identifying common themes and labeling significant issues in your dataset.
