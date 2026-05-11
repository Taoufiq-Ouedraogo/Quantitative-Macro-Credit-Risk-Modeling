"""
dataprep.py
"""



# Data manipulation
import numpy as np
import pandas as pd

# Scikit-learn base
from sklearn.base import BaseEstimator, TransformerMixin

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler



class LoanFeaturesPreparator(BaseEstimator, TransformerMixin):
    """
    Transform any new flow with the same rules
    Handles categorical encoding, numerical treatment, target preparation
    """

    def fit(self, df, y=None):
        # grade mapping
        grades = 'ABCDEFG'
        self.grade_map_ = {g: i+1 for i, g in enumerate(grades)}

        # sub_grade mapping
        n_subgrade = 5
        sub_grades = [f'{g}{n}' for g in grades for n in range(1, n_subgrade+1)]
        self.sub_grade_map_ = {sg: i+1 for i, sg in enumerate(sub_grades)}

        # purpose, home_ownership, verification_status mapping
        for col in ['purpose', 'home_ownership', 'verification_status']:
            dict_ = {
                cat: code for code, cat in enumerate(sorted(df[col].dropna().unique()))
            }
            setattr(self, f'{col}_map_', dict_)

        # addr_state frequency encoding
        self.state_freq_map_ = df['addr_state'].value_counts(normalize=True).to_dict()

        # Numerical columns treatment config
        self.binary_flag_cols_ = [
            'delinq_2yrs', 'pub_rec', 'pub_rec_bankruptcies'
        ]
        self.log_cols_ = [
            'installment', 'annual_inc', 'avg_cur_bal',
            'tot_cur_bal', 'revol_bal', 'open_acc'
        ]
        self.scale_cols_ = [
            'dti', 'total_acc', 'credit_age_months',
            'addr_state', 'int_rate'
        ] + self.log_cols_

        self.scaler_ = StandardScaler()
        self.imputer_ = SimpleImputer(strategy='median')
        return self


    def encode_vars(self, df_):
        df = df_.copy()
        for col in ['grade', 'sub_grade', 'purpose', 'home_ownership', 'verification_status']:
            # unseen categories = -1
            dict_ = getattr(self, f'{col}_map_')
            df[col] = df[col].map(dict_).fillna(-1).astype(int)

        df['term'] = df['term'].str.extract(r'(\d+)').astype(int)
        df['initial_list_status'] = (df['initial_list_status'] == 'w').astype(int)
        df['emp_length'] = (
            df['emp_length']
            .replace({'< 1 year': '0 years', '10+ years': '10 years'})
            .str.extract(r'(\d+)')[0]
            .fillna(0)
            .astype(float)
        )

        # frequency encoding, unseen states : global mean frequency
        mean_freq = np.mean(list(self.state_freq_map_.values()))
        df['addr_state'] = df['addr_state'].map(self.state_freq_map_).fillna(mean_freq)

        # credit age in months from loan issue date
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')
        df['credit_age_months'] = (
            (df['date'].dt.year  - df['earliest_cr_line'].dt.year)  * 12
          + (df['date'].dt.month - df['earliest_cr_line'].dt.month)
        ).astype(float)
        df = df.drop(columns=['earliest_cr_line'])
        return df



    def binary_flag_vars(self, df):
        df[self.binary_flag_cols_] = (df[self.binary_flag_cols_] > 0).astype(int)
        return df

    def log_scale_vars(self, df):
        df[self.log_cols_] = np.log1p(df[self.log_cols_].clip(lower=0))
        return df

    def standard_scale_vars(self, df):
        df[self.scale_cols_] = self.scaler_.transform(df[self.scale_cols_])
        return df

    def bins_loan_amnt(self, df):
        bins = [0, 5000, 10000, 15000, 20000, 25000, 35000, np.inf]
        labels = [1, 2, 3, 4, 5, 6, 7]
        df['loan_amnt'] = pd.cut(
            df['loan_amnt'], bins=bins, labels=labels, right=True
        ).astype(int)
        return df

    def impute_vars(self, df):
        cols_with_nan = df.columns[df.isna().any()].tolist()
        if len(cols_with_nan) > 0:
            df[cols_with_nan] = self.imputer_.fit_transform(df[cols_with_nan])
        return df

    def transform(self, df_):
        df = df_.copy()

        # Convert date
        df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
        df = df.rename(columns={'issue_d': 'date'})
        df = df.sort_values("date").reset_index(drop=True)

        df = self.encode_vars(df)

        # Binary target: 1 = default, 0 = repaid
        completed_statuses = ['Fully Paid', 'Charged Off']
        df = df[df['loan_status'].isin(completed_statuses)].reset_index(drop=True)
        df['default'] = (df['loan_status'] == 'Charged Off').astype(int)
        df = df.drop(columns=['loan_status'])

        # Numerical treatments
        self.scaler_.fit(df[self.scale_cols_])
        df = self.binary_flag_vars(df)
        df = self.log_scale_vars(df)
        df = self.standard_scale_vars(df)
        df = self.bins_loan_amnt(df)
        df = self.impute_vars(df)
        return df


def load_macro_data():
    """Load real macroeconomic data from FRED"""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    def get_fred_series(series_id):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url)
        df.columns = ["date", series_id]
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    # Retrieve key macro series
    fed = get_fred_series("DFF")                    # Fed Funds Rate
    unemp = get_fred_series("UNRATE")               # Unemployment rate
    cpi = get_fred_series("CPIAUCSL")               # Inflation index
    ind_prod = get_fred_series("INDPRO")            # Industrial production
    gdp = get_fred_series("GDP")                    # GDP
    credit_spread = get_fred_series("BAA10Y")       # Corporate risk premium
    treasury_10y = get_fred_series("GS10")       # Long-term rates
    treasury_2y = get_fred_series("GS2")       # Short-term rates

    # Merge all
    macro = fed.join([
        unemp, cpi, ind_prod, gdp, credit_spread, treasury_10y, treasury_2y
        ], how="outer")
    macro.columns = [
        "fed_funds", "unemp_rate", "cpi", "ind_prod", "real_gdp",
        "credit_spread", "treasury_10y", "treasury_2y"
    ]
    macro = macro.sort_index()

    # Forward-fill missing values
    macro = macro.ffill()
    return macro.reset_index()