import pandas as pd
import os


if not os.path.exists('data/loans_2007.csv'):
    print('converting new file')
    loans_2007 = pd.read_csv('data/LoanStats3a.csv', skiprows=1, low_memory=False)
    half_count = len(loans_2007) / 2
    loans_2007 = loans_2007.dropna(thresh=half_count, axis=1)
    loans_2007 = loans_2007.drop(['desc', 'url'], axis=1)
    loans_2007.to_csv('data/loans_2007.csv', index=False)
    del loans_2007

loans_2007 = pd.read_csv('data/loans_2007.csv', encoding='ISO-8859-1', low_memory=False)

drop = ['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'grade', 'sub_grade', 'emp_title', 'issue_d']

loans_2007.drop(drop, inplace=True, axis=1)

drop2 = ['zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt']
loans_2007.drop(drop2, inplace=True, axis=1)

drop3 = ['total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt']
loans_2007.drop(drop3, inplace=True, axis=1)

# Target
target = 'loan_status'

# Only keep Fully Paid and Charged Off and ignore other statues

drop4 = loans_2007[target].value_counts()[:2].index
loans_2007 = loans_2007[loans_2007[target].isin(drop4)]

loans_2007[target] = loans_2007[target].astype('category').cat.codes


data_mapping = dict()
for i in loans_2007.columns:
    data_mapping[i] = loans_2007[i].unique().shape[0]

print({k: v for k, v in data_mapping.items() if v < 2})
