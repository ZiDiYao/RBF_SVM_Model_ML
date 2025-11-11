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




    def cross_validation(self,k, X, Y):
        # CODE HERE !
        kf = KFold(n_splits=k,shuffle=True, random_state=42)
        Accuracy_list = []
        Recall_list = []
        Precision_list = []
        F1_score_list = []
        yhat_all = np.empty_like(Y)
        cm_sum = np.array([[0, 0], [0, 0]], dtype=int)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            clf = self.training(X_train,Y_train)
            yhat = clf.predict(X_test)
            yhat_all[test_idx]=yhat
            accuracy_val, recall_val, precision_val, F1_val = acc_recall_precision_F1(yhat, Y_test)
            Accuracy_list.append(accuracy_val)
            Recall_list.append(recall_val)
            Precision_list.append(precision_val)
            F1_score_list.append(F1_val)


            #print(f"Fold {fold} | TSS = {tss_val:.4f}")

        #input_array_display(k,input_array,arrayname)
            ######this method will plot figure corresponding to input array ########
        confusion_matrix_display(yhat_all,Y)
        roc_auc_display(yhat_all,Y)

        #print(f"TSS (mean ± std) over {k} folds: {mean_tss:.4f} ± {std_tss:.4f}")

        #output: rate of Accuracy, Recall, Precision, F1-score, AUC-ROC. 
        mean_array =float(np.mean(mean_array))
        mean_recall = float(np.mean(mean_recall))
        mean_precision = float(np.mean(mean_precision))
        mean_f1 = float(np.mean(mean_f1))
        return mean_array, mean_recall, mean_precision, mean_f1


    def acc_recall_precision_F1(self,Yhat,Y):
        # CODE HERE !
        y = np.asarray(Y).ravel()
        yhat = np.asarray(Yhat).ravel()
        TP = np.sum((y == 1) & (yhat == 1))
        TN = np.sum((y == 0) & (yhat == 0))
        FP = np.sum((y == 0) & (yhat == 1))
        FN = np.sum((y == 1) & (yhat == 0))
        tpr_den = TP + FN
        fpr_den = FP + TN
        #print(f"TP={TP},TN={TN},FP={FP},FN={FN}")
        accuracy = (TP+TN)/(TP+TN+FP+FN)


        if(FP+TP == 0 ):
            precision=0
        else:precision = (TP)/(TP+FP)
        if(TP+TN ==0):
            recall =0
        else:
            recall = TP / (TP + TN)
        if(precision+recall==0):
            F1=0
        else:
            F1 = 2 * (precision * recall) / (precision + recall)

        #print(f"accuracy={accuracy},precision={precision},recall={recall},F1={F1}")
        if tpr_den == 0 or fpr_den == 0: #avoid divided by zero
            #print(f"tpr_den:{tpr_den}, fpr_den:{fpr_den}")
            return 0.0

        return accuracy, recall, precision, F1


    def confusion_matrix_display(self,yhat,y):
        cm = confusion_matrix(y_pred=yhat,y_true=y,labels=[0,1])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Flare", "Flare"])
        disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
        ax.set_title(f"Type I and Type II error graph")
        plt.tight_layout()
        plt.show()
    
    def roc_auc_display(self,y_scores,y_true):
        auc = roc_auc_score(y_true, y_scores)
        print("AUC:", auc)

        # 绘制ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')  # 随机线
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()

    def input_array_display(self,k,input_array=[],arrayname):
        if input_array is None:
            print("Warning: input_array is empty.")
            return 0

        x = np.arange(1, k + 1)

        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.bar(x, input_array, color='skyblue', edgecolor='black', alpha=0.8)

        ax.set_title(f"{k}-Fold Input_array Scores ({featureSet})", fontsize=12, fontweight='bold')
        ax.set_xlabel("Fold", fontsize=10)
        ax.set_ylabel(f"{arrayname} Score", fontsize=10)
        ax.set_xticks(x)
        ax.set_ylim(0, max(input_array) * 1.1)

        # 在每个bar上标数值
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)

        plt.tight_layout()
        plt.show()
        return 0