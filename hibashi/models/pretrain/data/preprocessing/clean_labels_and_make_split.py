"""
Data cleaning and split the data into train, val and test for the  dataset
"""
import csv
from os import path as osp

import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

base_data_path = '/Users/elias/Downloads/fashion-dataset/'

#######################################################################################################################

print("Step 1: reformat malformed label entries ...")
# there are entries in the style.csv that have 11 or more columns instead of 10.
# after inspection, we see that there is an extra comma in the description.
with open(osp.join(base_data_path, 'styles.csv'), newline='', mode='r') as csv_in:
    with open(osp.join(base_data_path, 'styles_fixed.csv'), mode='w') as csv_out:
        csv_reader = csv.reader(csv_in, delimiter=',')
        csv_writer = csv.writer(csv_out, delimiter=',')
        for row in csv_reader:
            if len(row) > 10:
                row = row[:9] + [' - '.join(row[9:])]
            elif len(row) == 10:
                pass
            else:
                raise NotImplementedError

            csv_writer.writerow(row)

#######################################################################################################################

print("\nStep 2: Check if all files are present and remove labels without corresponding image ...")
df = pd.read_csv(osp.join(base_data_path, 'styles_fixed.csv'))
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)

print('Number of labels before filtering out images without corresponding images: ', len(df['image']))

row_idxs_to_remove = []
for i, row in df.iterrows():
    path = osp.join(base_data_path, 'images', row['image'])
    if not osp.exists(path):
        row_idxs_to_remove.append((i, row['image']))

df.drop([df.index[idx] for idx, filename in row_idxs_to_remove], inplace=True)

print('Number of labels after filtering out images without corresponding images: ', len(df['image']))


#######################################################################################################################

print("\nStep 3: Remove samples with non-sensical images ...")

print('Number of labels before filtering out bad images: ', len(df['image']))

bad_image_idxs = [44998]
df.drop([df.index[df['id'] == idx][0] for idx in bad_image_idxs], inplace=True)

print('Number of labels after filtering out bad images: ', len(df['image']))

#######################################################################################################################

print("\nStep 4: Split train and test according to year. Odd years -> test, even years -> train ...")

test_df = pd.concat([df.loc[df['year'] == year] for year in range(2007, 2020, 2)])
train_df = pd.concat([df.loc[df['year'] == year] for year in range(2008, 2020, 2)])

print('Number of training samples: ', len(train_df))
print('Number of test samples: ', len(test_df))

#######################################################################################################################

print("\nStep 5: Select the top 20 most occurring article types for pretraining ...")

top_20 = df['articleType'].value_counts().nlargest(20)

# sanity check if we have the same article types as in the challenge
article_types_pdf = ["Jeans", "Perfume and Body Mist", "Formal Shoes", "Socks", "Backpacks",
                     "Belts", "Briefs", "Sandals", "Flip Flops", "Wallets",
                     "Sunglasses", "Heels", "Handbags", "Tops", "Kurtas",
                     "Sports Shoes", "Watches", "Casual Shoes", "Shirts", "Tshirts"]
for article_type in top_20.index.values:
    if article_type not in article_types_pdf:
        print(article_type)
    assert article_type in article_types_pdf

all_article_types = df['articleType'].value_counts().index.values

pretrain_train_df = pd.concat([train_df.loc[train_df['articleType'] == at] for at in top_20.index.values])
pretrain_test_df = pd.concat([test_df.loc[test_df['articleType'] == at] for at in top_20.index.values])

print('pretrain-train, no. samples:', len(pretrain_train_df))
print('pretrain-test, no. samples:', len(pretrain_test_df))


print("\nStep 6: Select the rare article types for finetuning."
      "\nFilter out article types that are not present in both train and test, or in extreme low quantities in train,"
      "\n so that it prevents us from making a validation split for those categories ...")
finetune_train_before_filter = pd.concat(
    [train_df.loc[train_df['articleType'] == at] for at in all_article_types if at not in top_20.index.values])

finetune_test_before_filter = pd.concat(
    [test_df.loc[test_df['articleType'] == at] for at in all_article_types if at not in top_20.index.values])

print('finetune-train, no. samples before filtering:', len(finetune_train_before_filter))
print('finetune-test, no. samples before filtering:', len(finetune_test_before_filter))

low_occurence_types = []
for article_type in all_article_types:
    if article_type not in top_20.index.values:
        n_in_train = len(finetune_train_before_filter.loc[finetune_train_before_filter['articleType'] == article_type])
        n_in_test = len(finetune_test_before_filter.loc[finetune_test_before_filter['articleType'] == article_type])
        # removing categories that have an extremely low number of items
        if n_in_train < 3 or n_in_test < 2:
            low_occurence_types.append(article_type)
print('Removing low occurrence types:')
print(low_occurence_types)

finetune_train_df = pd.concat([train_df.loc[train_df['articleType'] == articleType] for articleType in all_article_types
                               if articleType not in top_20.index.values and articleType not in low_occurence_types])

finetune_test_df = pd.concat([test_df.loc[test_df['articleType'] == articleType] for articleType in all_article_types
                              if articleType not in top_20.index.values and articleType not in low_occurence_types])

#######################################################################################################################

print("\nStep 7: Make stratified train and validation splits ...")

y = pretrain_train_df['articleType'].to_frame()
pretrain_train, pretrain_val, _, _ = train_test_split(pretrain_train_df, y, stratify=y, test_size=0.2)

y = finetune_train_df['articleType'].to_frame()
finetune_train, finetune_val, _, _ = train_test_split(finetune_train_df, y, stratify=y, test_size=0.3)

# sanity check if the split was successful for pretraining split
train_counts = pretrain_train['articleType'].value_counts()
val_counts = pretrain_val['articleType'].value_counts().to_dict()
for train_cnt, atype in zip(train_counts, train_counts.index):
    val_cnt = val_counts[atype]
    assert 0.19 < val_cnt/(train_cnt+val_cnt) < 0.21

# sanity check if the split was successful for finetune split
train_counts = finetune_train['articleType'].value_counts()
val_counts = finetune_val['articleType'].value_counts().to_dict()
for train_cnt, atype in zip(train_counts, train_counts.index):
    val_cnt = val_counts[atype]
    # a bit more lenient here since we have categories with very few samples
    assert 0.15 < val_cnt/(train_cnt+val_cnt) < 0.45

# sanity check if all categories are present in each finetune set
assert set(finetune_train['articleType'].unique()) == \
       set(finetune_val['articleType'].unique()) == \
       set(finetune_test_df['articleType'].unique())

# printing out categories and their class idx
print('Categories and their class idx for pretrain_train, to be copied to model file:')
for idx, article_type in enumerate(sorted(pretrain_train['articleType'].unique())):
    print(f'"{article_type}": {idx},')
prin('Categories and their class idx for finetune_train_df, to be copied to model file:')
for idx, article_type in enumerate(sorted(finetune_train_df['articleType'].unique())):
    print(f'"{article_type}": {idx},')

#######################################################################################################################

print("\nStep 8: saving splits ...")

pretrain_train.to_pickle(osp.join(base_data_path, 'pretrain_train.df.pkl'))
pretrain_val.to_pickle(osp.join(base_data_path, 'pretrain_val.df.pkl'))
pretrain_test_df.to_pickle(osp.join(base_data_path, 'pretrain_test.df.pkl'))

finetune_train.to_pickle(osp.join(base_data_path, 'finetune_train.df.pkl'))
finetune_val.to_pickle(osp.join(base_data_path, 'finetune_val.df.pkl'))
finetune_test_df.to_pickle(osp.join(base_data_path, 'finetune_test.df.pkl'))
