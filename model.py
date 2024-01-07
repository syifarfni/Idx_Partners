import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore") 

#1. Pemahaman data
#membaca data
data=pd.read_csv('C:\IDX partners\datacredit.csv')
data.head()
#mengidentifikasi missing value
data.info()
data.loan_status.unique()

#2. menentukan target
data['good_bad']= np.where(data.loc[:, 'loan_status'].isin (['Charged Off','Default','Late (31-120 days)', 
                                                         'In Grace Period', 'Late (16-30 days)','Does not meet the credit policy. Status:Charged Off'])
                            , 1 , 0)
data.good_bad.value_counts()
data[['loan_status', 'good_bad']]

#3. data cleaning
missing_values = pd.DataFrame(data.isnull().sum()/data.shape[0])     
missing_values = missing_values [missing_values.iloc[:, 0]>0.50]
missing_values.sort_values([0], ascending=False)                   
# Menghapus kolom dengan lebih dari 50% nilai null
data.dropna(thresh=data.shape[0]*0.5, axis=1, inplace=True)
missing_values = pd.DataFrame(data.isnull().sum()/data.shape[0])
missing_values = missing_values [missing_values.iloc[:, 0]>0.50]
missing_values.sort_values([0], ascending=False)

#4. Data Splitting
from sklearn.model_selection import train_test_split
#membagi dataset menjadi 80% train dan 20% untuk test
X = data.drop('good_bad', axis=1)
y = data['good_bad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify= y, random_state=42)
X_train.shape, X_test.shape
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)
print(X_train)


#5. Data Prepocessing
for col in X_train.select_dtypes(include=['object', 'bool']).columns:
    print(col)
    print(X_train[col].unique())
    print()
# Kolom/feature yang harus di cleaning
col_need_to_clean = ['term', 'emp_length', 'issue_d', 'earliest_cr_line', 'last_pymnt_d', 
                    'next_pymnt_d', 'last_credit_pull_d']
X_train['term'].unique()
X_train['term'] = X_train['term'].astype(str)

# Gunakan metode .str.replace()
X_train['term'] = pd.to_numeric(X_train['term'].str.replace('months', ''))
X_train['term']
X_train['emp_length'].unique
# Convert the column to string type
X_train['emp_length'] = X_train['emp_length'].astype(str)

# Replace "10+ years" with "10"
X_train['emp_length'] = X_train['emp_length'].replace('10+ years', str(10))
# Remove " years"
X_train['emp_length'] = X_train['emp_length'].str.replace(' years', '')
# Replace "< 1 year" with "0"
X_train['emp_length'] = X_train['emp_length'].str.replace('< 1 year', str(0))
# Replace " year" with empty string
X_train['emp_length'] = X_train['emp_length'].str.replace(' year', '')

# Mengganti string kosong dengan "0"
X_train['emp_length'].replace('', '0', inplace=True)
# Mengonversi kolom 'emp_length' menjadi numerik
X_train['emp_length'] = pd.to_numeric(X_train['emp_length'], errors='coerce')
# Menangani nilai NaN setelah konversi
X_train['emp_length'].fillna(value=0, inplace=True)
# Mengonversi kolom 'emp_length' menjadi tipe data integer
X_train['emp_length'] = X_train['emp_length'].astype(int)

X_train['emp_length']
## Cek feature date
col_date = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
X_train['issue_d'].unique()
X_train['issue_d'] = pd.to_datetime(X_train['issue_d'], format='%b-%y', errors='coerce')
for col in col_date:
    X_train[col] = pd.to_datetime(X_train[col], errors='coerce', format='%b-%y')
X_train['issue_d']
#melakukan yang sama untuk X_test
X_test['term'] = X_test['term'].astype(str)

# Gunakan metode .str.replace()
X_test['term'] = pd.to_numeric(X_test['term'].str.replace('months', ''))

# Convert the column to string type
X_test['emp_length'] = X_test['emp_length'].astype(str)

# Replace "10+ years" with "10"
X_test['emp_length'] = X_test['emp_length'].replace('10+ years', str(10))
# Remove " years"
X_test['emp_length'] = X_test['emp_length'].str.replace(' years', '')
# Replace "< 1 year" with "0"
X_test['emp_length'] = X_test['emp_length'].str.replace('< 1 year', str(0))
# Replace " year" with empty string
X_test['emp_length'] = X_test['emp_length'].str.replace(' year', '')
# Mengganti string kosong dengan "0"
X_test['emp_length'].replace('', '0', inplace=True)
# Mengonversi kolom 'emp_length' menjadi numerik
X_test['emp_length'] = pd.to_numeric(X_test['emp_length'], errors='coerce')
# Menangani nilai NaN setelah konversi
X_test['emp_length'].fillna(value=0, inplace=True)
# Mengonversi kolom 'emp_length' menjadi tipe data integer
X_test['emp_length'] = X_test['emp_length'].astype(int)
X_test['issue_d'] = pd.to_datetime(X_test['issue_d'], format='%b-%y', errors='coerce')
for col in col_date:
    X_test[col] = pd.to_datetime(X_test[col], errors='coerce', format='%b-%y')
X_test['issue_d']
# Check apakah berhasil di cleaning
X_test[col_need_to_clean].info()
#feature engineering
# Kolom yang akan di feature engineering
col_need_to_clean
X_train[col_need_to_clean]
# tidak dibutuhkan untuk feature engineering
del X_train['next_pymnt_d']
del X_test['next_pymnt_d']
from datetime import date

date.today().strftime('%Y-%m-%d')
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd

def date_columns(df, column):
    today_date = pd.to_datetime(date.today().strftime('%Y-%m-%d'))
    df[column] = pd.to_datetime(df[column], format="%b-%y")
    
    # Calculate the number of months
    df['mths_since_' + column] = (today_date - df[column]).dt.days // 30
    
    # Drop the original date column
    df.drop(columns=[column], inplace=True)

# Example usage
date_columns(X_train, 'earliest_cr_line')
date_columns(X_train, 'issue_d')
date_columns(X_train, 'last_pymnt_d')
date_columns(X_train, 'last_credit_pull_d')
# apply to X_test
date_columns(X_test, 'earliest_cr_line')
date_columns(X_test, 'issue_d')
date_columns(X_test, 'last_pymnt_d')
date_columns(X_test, 'last_credit_pull_d')
X_test.isnull().sum()
X_train.isnull().sum()
#dilakukan pengisian pada nilai yang masih kosong dengan median
for col in ['mths_since_issue_d', 'mths_since_last_pymnt_d', 'mths_since_last_credit_pull_d']:
    X_train.fillna(X_train.median(), inplace=True)
    X_test.fillna(X_test.median(), inplace=True)
print(X_train.isnull().sum())

## 6. (a) Modelling logistic regression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
result = pd.DataFrame(list(zip(y_pred,y_test)), columns = ['y_pred', 'y_test'])
result.head()
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)


sns.heatmap(cm, annot=True, fmt='.0f', cmap=plt.cm.Blues)
plt.xlabel('y_pred')
plt.ylabel('y_test')

plt.show()
from sklearn.metrics import classification_report

# Evaluasi model pada data test
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
#kolom sebelah kiri untuk prediksi kelas 0 (berhasil bayar)
#kolom sebelah kanan untuk prediksi kelas 1 (gagal bayar)
model.predict_proba(X_test)
# memprediksi probability dan mengambil probability kelas positive
y_pred = model.predict_proba(X_test)[:, 1] 
y_pred
(y_pred > 0.5).astype(int)
plt.hist(y_pred)
#karena ini banyaknya di sekitaran 0.0 sampai 0.1 maka kita tidak bisa menetapkan thresholdnya 0.5, kita harus mencari terlebih dahulu best thresholdnya dengan true dan false positive rate
from sklearn. metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# youden j-statistic
j = tpr - fpr

ix = np.argmax(j)

best_thresh = thresholds[ix]
best_thresh
## hipotesanya: jika hasil prediksi loan ada diatas 0.097 kemungkinan besar dia akan gagal bayar
y_pred = model.predict_proba(X_test)[:, 1]
(y_pred > 0.097).astype(int)
best_thresh = 0.097
y_pred_binary = (y_pred > best_thresh).astype(int)
# Hitung confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

# Gambar heatmap confusion matrix
sns.heatmap(cm, annot=True, fmt='.0f', cmap=plt.cm.Blues)
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.show()

#6. (b) Evaluasi
from sklearn.metrics import classification_report

# Evaluasi model pada data test
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
coefficients = model.coef_
model.coef_
model.intercept_
df_coeff = pd.DataFrame(model.coef_, columns=X_train.columns)
df_coeff
X_train.head()



