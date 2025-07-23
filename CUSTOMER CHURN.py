import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
df = pd.read_csv(r"C:\Users\yashr\Downloads\archive (1)\Churn_Modelling.csv")
df.head()
df_train = df.copy()
df_train.head()
df_train.info()
df_train.isnull().sum()
df_train.shape
df_train.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace = True)
df_train.head()
df_train['Exited'].value_counts()
from sklearn.preprocessing import StandardScaler, OneHotEncoder
sc = StandardScaler()
df_train[['CreditScore', 'Balance', 'Age', 'EstimatedSalary']] = sc.fit_transform(
    df_train[['CreditScore', 'Balance', 'Age', 'EstimatedSalary']]
)

oe = OneHotEncoder(sparse_output=False, drop=None) 
encoded_data = oe.fit_transform(df_train[['Geography', 'Gender']])

encoded_df = pd.DataFrame(
    encoded_data,
    columns=oe.get_feature_names_out(['Geography', 'Gender']),
    index=df_train.index
)

df_train = pd.concat([df_train.drop(columns=['Geography', 'Gender']), encoded_df], axis=1)
df_train.head()
sns.pairplot(df_train)
plt.show()
corr_mat = df_train.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_mat, annot = True)
plt.show()
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
x = df_train.drop(columns=['Exited'])
y = df_train['Exited']

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 42, test_size=0.2)
rf = RandomForestClassifier(random_state=25, n_jobs=1)
rf.fit(x_train, y_train)
y_predict = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(accuracy)
print(classification_report(y_test, y_predict, target_names = ['Not Exited', 'Exited']))
report = classification_report(y_test, y_predict, target_names=['Not Exited', 'Exited'], output_dict=True)

report_df = pd.DataFrame(report).transpose()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:-3, :].T, annot=True, cmap='Blues', fmt='.2f')  # Skip 'accuracy', 'macro avg', etc.

plt.title('Classification Report Heatmap')
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.show()
predicted_output = rf.predict(x)

submission_df = pd.DataFrame({
    'id' : df['RowNumber'],
    'predicted_output' : predicted_output
})

submission_df.head()
