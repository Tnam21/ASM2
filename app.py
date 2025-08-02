import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

st.set_page_config(layout="wide")
st.title("📦 Order Data Analysis App")

# Đọc file CSV
try:
    df = pd.read_csv("order_data_with_issues.csv")
except FileNotFoundError:
    st.error("❌ Không tìm thấy file 'order_data_with_issues.csv'. Vui lòng tải lên file đúng tên.")
    st.stop()

# Xử lý dữ liệu
try:
    df['Quantity'] = df.groupby('ProductName')['Quantity'].transform(lambda x: x.fillna(x.mean().round()))
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], errors='coerce')
    df['DeliveryDate'] = df['DeliveryDate'].fillna(df['OrderDate'] + timedelta(days=7))

    df = df.drop_duplicates(subset='OrderID', keep='first')
    df.loc[df['Quantity'] < 0, 'Quantity'] = df['Quantity'].abs()

    price_stats = df.groupby('ProductName')['UnitPrice'].agg(['mean', 'std']).reset_index()
    price_stats['lower_bound'] = price_stats['mean'] - 2 * price_stats['std']
    price_stats['upper_bound'] = price_stats['mean'] + 2 * price_stats['std']

    def fix_unit_price(row):
        product = row['ProductName']
        price = row['UnitPrice']
        stats = price_stats[price_stats['ProductName'] == product].iloc[0]
        if pd.isna(price) or price < stats['lower_bound'] or price > stats['upper_bound']:
            return stats['mean']
        return price

    df['UnitPrice'] = df.apply(fix_unit_price, axis=1)
    df.loc[df['DeliveryDate'] < df['OrderDate'], 'DeliveryDate'] = df['OrderDate'] + timedelta(days=7)

    df['Quantity'] = df['Quantity'].round().astype(int)
    df['UnitPrice'] = df['UnitPrice'].round(2)
    df['OrderDate'] = df['OrderDate'].dt.strftime('%Y-%m-%d')
    df['DeliveryDate'] = df['DeliveryDate'].dt.strftime('%Y-%m-%d')

except Exception as e:
    st.error(f"🚨 Lỗi xử lý dữ liệu: {e}")
    st.stop()

# Hiển thị dữ liệu
st.subheader("📋 Dữ liệu sau xử lý")
st.dataframe(df.head())

# Tạo cột Failure linh hoạt
threshold = df['Quantity'].median()
df['Failure'] = (df['Quantity'] > threshold).astype(int)

# Biểu đồ 1: Failure Distribution
st.subheader("📊 1. Phân phối Failure")
fig1, ax1 = plt.subplots()
sns.countplot(x='Failure', data=df, ax=ax1)
st.pyplot(fig1)

# Biểu đồ 2: Boxplot Quantity theo Product
st.subheader("📊 2. Phân phối Quantity theo Product")
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.boxplot(x='ProductName', y='Quantity', data=df, ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)

# Biểu đồ 3: Barplot Unit Price
st.subheader("📊 3. Giá trung bình theo Product")
fig3, ax3 = plt.subplots(figsize=(12, 6))
sns.barplot(x='ProductName', y='UnitPrice', data=df, ax=ax3)
plt.xticks(rotation=45)
st.pyplot(fig3)

# Biểu đồ 4: Order Volume Over Time
st.subheader("📊 4. Số lượng đơn hàng theo thời gian")
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
orders_by_date = df['OrderDate'].dt.date.value_counts().sort_index()
fig4, ax4 = plt.subplots(figsize=(10, 5))
orders_by_date.plot(kind='line', ax=ax4)
ax4.set_xlabel("Date")
ax4.set_ylabel("Số đơn hàng")
st.pyplot(fig4)

# Biểu đồ 5: Correlation Heatmap
st.subheader("📊 5. Ma trận tương quan")
fig5, ax5 = plt.subplots(figsize=(8, 5))
sns.heatmap(df[['Quantity', 'UnitPrice', 'Failure']].corr(), annot=True, cmap='Blues', ax=ax5)
st.pyplot(fig5)

# Mô hình Random Forest
st.subheader("🤖 6. Dự đoán Failure bằng Random Forest")

try:
    X = df[['Quantity', 'UnitPrice']]
    y = df['Failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    unique_labels = y_train.unique()
    st.write("🎯 Nhãn trong tập huấn luyện:", unique_labels)

    if len(unique_labels) < 2:
        st.warning("⚠️ Không đủ dữ liệu đa dạng để huấn luyện mô hình (chỉ có 1 nhãn duy nhất).")
    else:
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba") and model.predict_proba(X_test).shape[1] > 1:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)

            fig6, ax6 = plt.subplots()
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            ax6.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
            ax6.plot([0, 1], [0, 1], 'k--')
            ax6.set_title("ROC Curve")
            ax6.set_xlabel("False Positive Rate")
            ax6.set_ylabel("True Positive Rate")
            ax6.legend()
            st.pyplot(fig6)

            st.text("📄 Classification Report:")
            st.text(classification_report(y_test, y_pred))
        else:
            st.warning("⚠️ Không thể tính AUC vì model không trả về xác suất hai lớp.")
except Exception as e:
    st.error(f"🚨 Lỗi trong huấn luyện mô hình: {e}")
