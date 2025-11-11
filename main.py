import pandas as pd
from pathlib import Path

# ===== 1) 读取数据 =====
DATA_PATH = "dataSet/heart.csv"   # 改成你的文件名
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# ===== 2) 基本检查 =====
expected = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
            'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope','HeartDisease']
missing = [c for c in expected if c not in df.columns]
if missing:
    raise ValueError(f"缺少列: {missing}")

# ===== 3) 目标变量 y =====
y = df['HeartDisease'].astype(int) #*********这个就是处理好的y**********

# ===== 4) 特征 X（先复制一份）=====
X = df.drop(columns=['HeartDisease']).copy()

# ===== 5) 统一二元列到 0/1 =====
# Sex: M/F -> 1/0
if X['Sex'].dtype == object:
    X['Sex'] = X['Sex'].str.strip().map({'M': 1, 'F': 0}).astype(int)

# ExerciseAngina: Y/N -> 1/0
if X['ExerciseAngina'].dtype == object:
    X['ExerciseAngina'] = X['ExerciseAngina'].str.strip().map({'Y': 1, 'N': 0}).astype(int)

# FastingBS 已经是 0/1，但确保是 int
X['FastingBS'] = X['FastingBS'].astype(int)

# ===== 6) 对多类别字符串列做 one-hot =====
cat_cols = []
for c in ['ChestPainType', 'RestingECG', 'ST_Slope']:
    if c in X.columns and X[c].dtype == object:
        cat_cols.append(c)

X_encoded = pd.get_dummies(X, columns=cat_cols, prefix=cat_cols, drop_first=False)  #*****这个就是X处理好的可以直接用******


# ===== 7) 输出保存 =====
out_dir = Path("processed")
out_dir.mkdir(exist_ok=True)

X_path = out_dir / "X_encoded.csv"
y_path = out_dir / "y.csv"
cols_path = out_dir / "feature_names.txt"

X_encoded.to_csv(X_path, index=False)
y.to_csv(y_path, index=False, header=['HeartDisease'])

with open(cols_path, "w", encoding="utf-8") as f:
    for c in X_encoded.columns:
        f.write(c + "\n")

print("✅ 数据清洗与编码完成")
print(f"X_encoded 形状: {X_encoded.shape} -> {X_path}")
print(f"y 形状        : {y.shape} -> {y_path}")
print("特征列已保存到:", cols_path)

