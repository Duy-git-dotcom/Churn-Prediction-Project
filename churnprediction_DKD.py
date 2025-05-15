##---Các thư viện cần thiết cho bài toán dự đoán khả năng rời bỏ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

#---EDA-Khai phá dữ liệu:
#Nhập dữ liệu
pd.set_option('display.max_columns', None)
df = pd.read_csv(r"E:\AI_Learning\venv_demo\dataset_Telco\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df_filtered = df.drop(columns=['customerID'])
df_filtered['TotalCharges'] = df_filtered['TotalCharges'].replace(" ","0.0")
df_filtered['TotalCharges'] = df_filtered['TotalCharges'].astype(float)

#Sanity Check
def sanity_check(x):
    head = df_filtered.head(2)
    info = df_filtered.info()
    shape = df_filtered.shape
    column = df_filtered.columns
    describe = df_filtered.describe()
    print(x)

#Xem qua các giá trị độc nhất của các cột categorial
def unique_check():
    numerical_data = ['tenure','TotalCharges','MonthlyCharges']
    for col in df_filtered.columns:
        if col not in numerical_data:
            print(col, df[col].unique())
            print('-'*50)

#Kiểm tra phân bổ dữ liệu của các cột
def distribution_check():
    print('Churn distribution',df_filtered['Churn'].value_counts())
    print('Gender distribution',df_filtered['gender'].value_counts())
    print('Senior Citizen',df_filtered['SeniorCitizen'].value_counts())
    print('Partner',df_filtered['Partner'].value_counts())
    print('Phone Service',df_filtered['PhoneService'].value_counts())
    print('Dependents',df_filtered['Dependents'].value_counts())

#---Trực quan hóa và phân tích bằng biểu đồ:
#Phân bổ dữ liệu-histogram
def plot_histogram(dataframe, column):
    plt.figure(figsize=(8,5))
    sns.histplot(dataframe[column], kde=True)
    plt.title(f'Độ phân tán của {column}')
    col_mean= dataframe[column].mean()
    col_median= dataframe[column].median()
    plt.axvline(col_mean, color = 'green', linestyle= '-', label = 'Mean')
    plt.axvline(col_median, color = 'red', linestyle= '--', label = 'Median')
    plt.legend()
    plt.show()

#Phân bổ dữ liệu-boxplot:
def boxplot(dataframe, column):
    plt.figure(figsize=(5,8))
    sns.boxplot(y=dataframe[column])
    plt.title(f'Distribution of {column}')
    plt.ylabel(column)
    plt.show()
    
#Heatmap mối tương quan của 3 cột dữ liệu
def heatmap_corr(dataframe):
    plt.figure(figsize=(8,5))
    sns.heatmap(dataframe[['tenure','TotalCharges','MonthlyCharges']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation heatmap')
    plt.show()

#Countplot cho dữ liệu dạng phân loại
def cate_countplot(dataframe):
    object_cols= dataframe.select_dtypes(include='object').columns.to_list()
    object_cols= ['SeniorCitizen']+ object_cols

    for col in object_cols:
        plt.figure(figsize=(8,5))
        sns.countplot(x=dataframe[col])
        plt.title(f'Countplot of {col}')
    plt.show()

#---Tiền xử lý dữ liệu với LabelEncoder và lưu vào file pkl
encoder={}
object_cols= df_filtered.select_dtypes(include='object').columns 
for col in object_cols:
    le=LabelEncoder()
    df_filtered[col] = le.fit_transform(df_filtered[col])
    encoder[col] = le
    
with open('encoder', 'wb') as f:
    pickle.dump(encoder, f)

#---Chia bộ dữ liệu Train, Test với Churn là biến phụ thuộc
x = df_filtered.drop(columns=['Churn'])
y = df_filtered['Churn']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
#print(y_train.value_counts())#->dữ liệu không cân bằng, do chỉ có 6000 bản ghi, ta thực hiện oversampling

smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train,y_train)

#---Huấn luyện cây quyết định
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest":RandomForestClassifier(random_state=42),
    "XGboost":XGBClassifier(random_state=42)
}
#Kiểm định chéo-->RandomForest là mô hình cho thấy độ chính xác cao nhất
cv_score = {}
for model_name, model in models.items():
    print(f'Huấn luyện mô hình {model_name} với tham số mẫu')
    score = cross_val_score(model, x_train_smote, y_train_smote, cv=5, scoring='accuracy')
    cv_score[model_name] = score
    print(f'{model_name} có độ chính xác cross_validation:{np.mean(score):.2f}')
    print('--'*50)
    
rf = RandomForestClassifier(random_state=42)
#Training
rf.fit(x_train_smote,y_train_smote)
#Evaluating
y_test_prediction = rf.predict(x_test)
#Accuracy_point, Confusion_Matrix, Classification_Report
print('Độ chính xác:\n', accuracy_score(y_test,y_test_prediction))
print('Confusion Matrix:\n', confusion_matrix(y_test,y_test_prediction))
print('Classification Report:\n', classification_report(y_test,y_test_prediction))

#Lưu trọng số, định dạng vào file pickle
model_data = {'model': rf, 'feature_names': x.columns.to_list()}
with open ('model_weights.pkl', 'wb') as f:
    pickle.dump(model_data, f)







