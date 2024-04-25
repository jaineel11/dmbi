import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data = {'TransactionID': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        'Item': ['A', 'B', 'D', 'A', 'C', 'B', 'C', 'D', 'B', 'E']}
df = pd.DataFrame(data)

df.to_csv('transactions.csv', index=False)

df = pd.read_csv('transactions.csv')

transactions = df.groupby(['TransactionID', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('TransactionID')

transactions = transactions.applymap(lambda x: 1 if x >= 1 else 0)

frequent_itemsets = apriori(transactions, min_support=0.1, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)