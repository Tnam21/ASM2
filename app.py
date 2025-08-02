# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta

# Đọc dữ liệu gốc
df = pd.read_csv('order_data_with_issues.csv')

# Tiền xử lý dữ liệu như trong file bạn đã viết
# (... bạn copy phần xử lý vào đây ...)

# Sau khi xử lý
st.title("📊 Data Cleaning and Analysis App")

st.subheader("1. Cleaned Data Preview")
st.dataframe(df.head())

# Biểu đồ 1: Phân phối Failure
st.subheader("2. Failure Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='Failure', data=df, ax=ax1)
st.pyplot(fig1)

# Biểu đồ 2: Boxplot Quantity theo Product
st.subheader("3. Product Quantity Distribution")
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.boxplot(x='ProductName', y='Quantity', data=df, ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)

# Biểu đồ 3: Barplot Unit Price
st.subheader("4. Average Unit Price by Product")
fig3, ax3 = plt.subplots(figsize=(12, 6))
sns.barplot(x='ProductName', y='UnitPrice', data=df, ax=ax3)
plt.xticks(rotation=45)
st.pyplot(fig3)

# Biểu đồ 4: Orders over time
st.subheader("5. Order Volume Over Time")
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
orders_by_date = df['OrderDate'].dt.date.value_counts().sort_index()
fig4, ax4 = plt.subplots(figsize=(10, 5))
orders_by_date.plot(kind='line', ax=ax4)
ax4.set_title('Order Volume Over Time')
st.pyplot(fig4)

# Biểu đồ 5: Correlation
st.subheader("6. Correlation Heatmap")
fig5, ax5 = plt.subplots(figsize=(8, 5))
sns.heatmap(df[['Quantity', 'UnitPrice', 'Failure']].corr(), annot=True, cmap='Blues', ax=ax5)
st.pyplot(fig5)

# Machine learning: Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

X = df[['Quantity', 'UnitPrice']]
y = df['Failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    st.subheader("7. ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig6, ax6 = plt.subplots(figsize=(6, 6))
    ax6.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    ax6.plot([0, 1], [0, 1], 'k--')
    ax6.set_title('ROC Curve')
    ax6.legend()
    st.pyplot(fig6)

    st.subheader("8. Classification Report")
    st.text(classification_report(y_test, y_pred))
