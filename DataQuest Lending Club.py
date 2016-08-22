import pandas as pd
import os


if not os.path.exists('data/loans_2007.csv'):
    print('converting new file')
    loans = pd.read_csv('data/LoanStats3a.csv', skiprows=1, low_memory=False)
    half_count = len(loans) / 2
    loans = loans.dropna(thresh=half_count, axis=1)
    loans = loans.drop(['desc', 'url'], axis=1)
    loans.to_csv('data/loans_2007.csv', index=False)
    del loans

loans = pd.read_csv('data/loans_2007.csv', encoding='ISO-8859-1', low_memory=False)

drop = ['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'grade', 'sub_grade', 'emp_title', 'issue_d']

loans.drop(drop, inplace=True, axis=1)

drop2 = ['zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt']
loans.drop(drop2, inplace=True, axis=1)

drop3 = ['total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt']
loans.drop(drop3, inplace=True, axis=1)

# Target
target = 'loan_status'

# Only keep Fully Paid and Charged Off and ignore other statues

drop4 = loans[target].value_counts()[:2].index
loans = loans[loans[target].isin(drop4)]

loans[target] = loans[target].astype('category').cat.codes

# Removing columns with just one variable as they do not affect model

data_mapping = dict()
for i in loans.columns:
    data_mapping[i] = loans[i].dropna().unique().shape[0]

loans.drop([k for k, v in data_mapping.items() if v == 1], inplace=True, axis=1)

# Removing columns with too many missing values and remove rows with (any) NaN

null_counts = loans.isnull().sum()

loans.drop(['pub_rec_bankruptcies'], inplace=True, axis=1)
loans.dropna(inplace=True)

object_columns_df = loans.columns[loans.dtypes == 'object']

cols = ['home_ownership', 'verification_status', 'term', 'purpose']

# convert employment length to continuous

emp_length_tranform = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}

loans.replace(emp_length_tranform, inplace=True)

# convert interest rates to numeric

loans[['revol_util', 'int_rate']] = loans[['revol_util', 'int_rate']].apply(lambda x: x.str.replace('%',''), axis=0).astype(float)

cols_drop = ['last_credit_pull_d', 'addr_state', 'title', 'earliest_cr_line']

loans.drop(cols_drop, inplace=True, axis=1)


def transform_multinominal(df, cols):
    df = pd.concat([df, pd.get_dummies(df[cols])], axis=1)
    df.drop(cols, inplace=True, axis=1)
    return df

loans = transform_multinominal(loans, cols)
print(loans.columns)

