import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# =======================================================
# 1. 数据加载与预处理 (保持与之前一致)
# =======================================================
# 假设 df 是你加载好的 DataFrame
# df = pd.read_csv('your_data.csv') 

df = pd.read_excel('df.xlsx')

print(f"--- Step 2: Running Hybrid Random Forest Baseline ---")
print(f"数据总条数: {len(df)}")

# --- A. 硬特征构建 (Hard Features) ---
print("1. 构建硬特征 (Hard Features)...")

# 类型转换
df['x_duration'] = pd.to_numeric(df['x_duration'], errors='coerce').fillna(0)
df['x_price'] = pd.to_numeric(df['x_price'], errors='coerce').fillna(0)
df['x_pricedecreaserate'] = pd.to_numeric(df['x_pricedecreaserate'], errors='coerce').fillna(0)

# 计算项目统计量
proj_stats = df.groupby('bid_ann_guid')['x_price'].agg(['mean', 'std']).reset_index()
proj_stats.columns = ['bid_ann_guid', 'proj_mean', 'proj_std']
df = df.merge(proj_stats, on='bid_ann_guid', how='left')

# 构造核心衍生特征
df['feat_cv'] = df['proj_std'] / (df['proj_mean'] + 1e-6)
df['feat_dev'] = (df['x_price'] - df['proj_mean']) / (df['proj_mean'] + 1e-6)

hard_cols = ['feat_cv', 'feat_dev', 'x_price', 'x_pricedecreaserate', 'x_duration']

# --- B. 文本特征构建 (TF-IDF + SVD) ---
print("2. 构建文本特征 (TF-IDF + SVD)...")

# 拼接文本
df['text_raw'] = "供应商:" + df['x_providername'].fillna('') + " 联系人:" + df['x_employee'].fillna('')

# 1. TF-IDF 向量化
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2)) # 包含双词组合
tfidf_matrix = tfidf.fit_transform(df['text_raw'].astype(str))

# 2. SVD 降维 (Latent Semantic Analysis)
# 将高维稀疏矩阵压缩为 50 维稠密向量，模拟 Embedding
svd = TruncatedSVD(n_components=50, random_state=42)
text_features = svd.fit_transform(tfidf_matrix)

# 转为 DataFrame
text_cols = [f'svd_{i}' for i in range(text_features.shape[1])]
X_text = pd.DataFrame(text_features, columns=text_cols)
X_hard = df[hard_cols]

# --- C. 拼接 ---
X_hybrid = pd.concat([X_hard.reset_index(drop=True), X_text.reset_index(drop=True)], axis=1)
y = df['x_isqualified'] # Label

# =======================================================
# 2. 数据划分 (必须与 XGBoost 实验的 Random State 一致)
# =======================================================
print("3. 数据划分 (70/10/20)...")

# 第一刀：切出 70% 训练集
X_train, X_temp, y_train, y_temp = train_test_split(
    X_hybrid, y, test_size=0.30, random_state=42, stratify=y
)
# 第二刀：剩下的切成 Validation 和 Test (虽然这里只用 Test, 但为了保持数据一致性必须切)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=(2/3), random_state=42, stratify=y_temp
)

# =======================================================
# 3. 训练 Random Forest (作为 Classifier)
# =======================================================
print("4. 训练 Random Forest 分类器...")

# 初始化随机森林
# n_estimators=200: 树的数量
# class_weight='balanced': 处理类别不平衡
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1 # 使用所有 CPU 核心
)

# 训练
rf_clf.fit(X_train, y_train)

# =======================================================
# 4. 评估与结果输出
# =======================================================
# 预测
y_pred_proba = rf_clf.predict_proba(X_test)[:, 1]
y_pred_bin = rf_clf.predict(X_test)

auc_score = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred_bin)
acc = accuracy_score(y_test, y_pred_bin)

print("\n" + "="*40)
print("   Baseline Step 2: Random Forest (Hybrid)   ")
print("="*40)
print(f"AUC       : {auc_score:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"Accuracy  : {acc:.4f}")
print("-" * 40)
print("Interpretation:")
if auc_score < 0.8:
    print("结果符合预期：即使使用了文本特征，传统分类器(RF)依然无法达到 LLM 的推理精度。")
    print("这证明了 LLM 捕捉到了关键词之外的逻辑异常。")
else:
    print("注意：如果此分数非常高，说明硬特征泄露了太多信息，或者文本中包含直接的作弊证据。")