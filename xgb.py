import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score

# =======================================================
# 1. 读取并解析3个工作表的JSON数据
# =======================================================
print("--- Step 1: Read and Parse 3 Sheets ---")
excel_path = "all_testdata_origin.xlsx"

# 1.1 读取【项目信息】表（索引0）
print("正在读取【项目信息】表（索引0）...")
df_project_raw = pd.read_excel(excel_path, sheet_name=0)
def parse_project_json(row):
    try:
        json_str = row['项目JSON']
        if pd.isna(json_str) or json_str.strip() == "":
            return pd.Series([None, None])
        data = json.loads(json_str)
        return pd.Series([
            data.get('projguid', None),
            data.get('projname', "未知项目")
        ])
    except Exception as e:
        print(f"项目信息JSON解析错误（行{row.name}）: {e}")
        return pd.Series([None, None])
df_project_parsed = df_project_raw.apply(parse_project_json, axis=1)
df_project_parsed.columns = ['projguid', 'projname']
# 放宽过滤：只保留有projguid的数据
df_project = df_project_parsed.dropna(subset=['projguid'])
print(f"【修改后】项目信息表有效行数: {len(df_project)}")

# 1.2 读取【招标公告】表（索引1）
print("\n正在读取【招标公告】表（索引1）...")
df_bid_notice_raw = pd.read_excel(excel_path, sheet_name=1)
def parse_notice_json(row):
    try:
        json_str = row['招标JSON']
        if pd.isna(json_str) or json_str.strip() == "":
            return pd.Series([None, None, None])
        data = json.loads(json_str)
        return pd.Series([
            data.get('bid_ann_guid', None),
            data.get('projguid', None),
            data.get('bidnoticetitle', "未知招标")
        ])
    except Exception as e:
        print(f"招标公告JSON解析错误（行{row.name}）: {e}")
        return pd.Series([None, None, None])
df_notice_parsed = df_bid_notice_raw.apply(parse_notice_json, axis=1)
df_notice_parsed.columns = ['bid_ann_guid', 'projguid', 'bidnoticetitle']
# 放宽过滤：只保留有bid_ann_guid的数据（允许projguid为空）
df_notice = df_notice_parsed.dropna(subset=['bid_ann_guid'])
print(f"【修改后】招标公告表有效行数: {len(df_notice)}")

# 1.3 读取【投标商数据】表（索引2）
print("\n正在读取【投标商数据】表（索引2）...")
df_bidder_raw = pd.read_excel(excel_path, sheet_name=2)
def parse_bidder_json(row):
    try:
        json_str = row['投标JSON']
        if pd.isna(json_str) or json_str.strip() == "":
            return pd.Series([None]*7)
        data = json.loads(json_str)
        return pd.Series([
            data.get('bid_ann_guid', None),
            data.get('x_providername', None),
            data.get('x_employee', None),
            data.get('x_price', None),
            data.get('x_duration', None),
            data.get('x_pricedecreaserate', None),
            data.get('x_isqualified', None)
        ])
    except Exception as e:
        print(f"投标商JSON解析错误（行{row.name}）: {e}")
        return pd.Series([None]*7)
df_bidder_parsed = df_bidder_raw.apply(parse_bidder_json, axis=1)
df_bidder_parsed.columns = ['bid_ann_guid', 'x_providername', 'x_employee', 
                            'x_price', 'x_duration', 'x_pricedecreaserate', 'x_isqualified']
# 放宽过滤：只保留有bid_ann_guid和x_isqualified的数据
df_bidder = df_bidder_parsed.dropna(subset=['bid_ann_guid', 'x_isqualified'])
print(f"【修改后】投标商数据表有效行数: {len(df_bidder)}")

# =======================================================
# 2. 三表关联（修复关联逻辑：左连接+格式统一）
# =======================================================
print("\n--- Step 2: Merge 3 Tables (Fixed) ---")
# 第一步：招标公告 ↔ 项目信息（左连接，保留招标表所有数据）
df_notice_project = pd.merge(
    df_notice, df_project, 
    on='projguid', 
    how='left'  # 关键：inner→left，避免项目表数据不全导致关联为空
)
print(f"招标+项目左连接后行数: {len(df_notice_project)}")

# 第二步：统一bid_ann_guid格式（转字符串+去空格，解决格式不匹配）
print("\n【统一关联字段格式】")
df_notice_project['bid_ann_guid'] = df_notice_project['bid_ann_guid'].astype(str).str.strip()
df_bidder['bid_ann_guid'] = df_bidder['bid_ann_guid'].astype(str).str.strip()

# 调试：查看匹配情况
match_count = df_bidder[df_bidder['bid_ann_guid'].isin(df_notice_project['bid_ann_guid'])].shape[0]
print(f"投标数据中，与招标表匹配的行数: {match_count}")

# 第三步：关联投标商数据（仍用inner，确保有对应招标数据）
df_final = pd.merge(
    df_bidder, df_notice_project, 
    on='bid_ann_guid', 
    how='inner'
)
print(f"三表关联后最终有效行数: {len(df_final)}")

# 紧急方案：如果仍为空，只关联投标和招标表（跳过项目表）
if len(df_final) == 0:
    print("\n【紧急方案】三表关联为空，改用投标+招标表关联")
    df_final = pd.merge(
        df_bidder, df_notice, 
        on='bid_ann_guid', 
        how='inner'
    )
    print(f"投标+招标关联后行数: {len(df_final)}")
    # 补充项目名称字段（避免后续代码报错）
    if 'projname' not in df_final.columns:
        df_final['projname'] = "未知项目"

# 最终检查：如果仍为空，报错并提示
if len(df_final) == 0:
    raise ValueError("所有关联方式均为空！请检查三个表的bid_ann_guid/projguid是否有重合值。")

# =======================================================
# 3. 数据预处理与硬特征工程
# =======================================================
print("\n--- Step 3: Data Preprocessing & Hard Feature Engineering ---")

# 3.1 身份脱敏
print("正在执行供应商/联系人脱敏...")
provider_unique = df_final['x_providername'].dropna().unique()
provider_mapping = {name: f"Provider_{i}" for i, name in enumerate(provider_unique)}
employee_unique = df_final['x_employee'].dropna().unique()
employee_mapping = {name: f"Employee_{i}" for i, name in enumerate(employee_unique)}
df_final['anon_provider'] = df_final['x_providername'].map(provider_mapping).fillna('Unknown_Provider')
df_final['anon_employee'] = df_final['x_employee'].map(employee_mapping).fillna('Unknown_Employee')

# 3.2 硬特征工程
print("正在构建硬数值特征...")
df_final['x_price'] = pd.to_numeric(df_final['x_price'], errors='coerce').fillna(0)
df_final['x_duration'] = pd.to_numeric(df_final['x_duration'], errors='coerce').fillna(0)
df_final['x_pricedecreaserate'] = pd.to_numeric(df_final['x_pricedecreaserate'], errors='coerce').fillna(0)

# 按招标ID分组计算统计量
bid_price_stats = df_final.groupby('bid_ann_guid')['x_price'].agg(['mean', 'std', 'count']).reset_index()
bid_price_stats.columns = ['bid_ann_guid', 'bid_price_mean', 'bid_price_std', 'bid_count']
df_final = pd.merge(df_final, bid_price_stats, on='bid_ann_guid', how='left')

# 计算特征
df_final['feat_price_cv'] = df_final['bid_price_std'] / (df_final['bid_price_mean'] + 1e-6)
df_final['feat_price_dev'] = (df_final['x_price'] - df_final['bid_price_mean']) / (df_final['bid_price_mean'] + 1e-6)
df_final['feat_bid_count_per_ann'] = df_final['bid_count']

# 特征列
hard_feature_cols = [
    'feat_price_cv', 'feat_price_dev', 'feat_bid_count_per_ann',
    'x_price', 'x_duration', 'x_pricedecreaserate'
]

# =======================================================
# 4. 模型输入准备+数据划分（此时数据已非空）
# =======================================================
print("\n--- Step 4: Prepare Model Input & Split Data ---")
X_hybrid = df_final[hard_feature_cols].copy().fillna(0)
y = df_final['x_isqualified'].astype(int)

print(f"最终特征矩阵维度: {X_hybrid.shape}")
print(f"标签分布（0=正常: {sum(y==0)}, 1=围标: {sum(y==1)}）")

# 数据划分（此时n_samples>0，不会报错）
X_train, X_temp, y_train, y_temp = train_test_split(
    X_hybrid, y, test_size=0.3, random_state=42, stratify=y if sum(y) > 0 else None
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp if sum(y_temp) > 0 else None
)

print(f"训练集: {len(X_train)} 条 | 验证集: {len(X_val)} 条 | 测试集: {len(X_test)} 条")

# =======================================================
# 5. 训练XGBoost模型+评估
# =======================================================
print("\n--- Step 5: Train XGBoost & Evaluate ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=120, learning_rate=0.06, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
    objective='binary:logistic', use_label_encoder=False, random_state=42
)

# 训练模型
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=15,
    verbose=10
)

# 评估
y_test_proba = xgb_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= 0.5).astype(int)
test_auc = roc_auc_score(y_test, y_test_proba) if len(set(y_test)) > 1 else 0.0
test_f1 = f1_score(y_test, y_test_pred) if len(set(y_test)) > 1 else 0.0
test_recall = recall_score(y_test, y_test_pred) if len(set(y_test)) > 1 else 0.0

print("\n" + "=" * 60)
print("【围标串标识别模型 - 测试集评估结果】")
print(f"AUC值          : {test_auc:.4f}")
print(f"F1分数         : {test_f1:.4f}")
print(f"召回率         : {test_recall:.4f}")
print("=" * 60)

# =======================================================
# 6. 保存结果
# =======================================================
print("\n--- Step 6: Save Results ---")
test_result_df = pd.DataFrame({
    'bid_ann_guid': df_final.loc[X_test.index, 'bid_ann_guid'].values,
    'x_providername': df_final.loc[X_test.index, 'x_providername'].values,
    'x_price': df_final.loc[X_test.index, 'x_price'].values,
    'true_label': y_test.values,
    'xgb_proba': y_test_proba
})
test_result_df.to_excel("围标识别结果_修复关联.xlsx", index=False)
xgb_model.save_model("围标识别XGB模型_修复关联.json")

print("\n✅ 结果已保存！")
print("🎉 所有流程执行完成！")