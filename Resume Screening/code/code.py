# Resume Screening ML Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import pandas as pd
import os
# =======================================================================
# LOAD DATASET
# =======================================================================

df = pd.read_csv("Resume Screening.csv")
X = df["Resume"]
y = df["Category"]

# =======================================================================
# TF-IDF FEATURE EXTRACTION
# =======================================================================

tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# =======================================================================
# SUPERVISED MODELS
# =======================================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Linear SVM": LinearSVC(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": MultinomialNB()
}

metrics = []

# =======================================================================
# TRAIN MODELS + CONFUSION MATRICES
# =======================================================================

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec = recall_score(y_test, preds, average='macro', zero_division=0)

    metrics.append([name, acc, prec, rec])

    print(f"{name} Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=False, cmap='viridis')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"cm_{name.replace(' ', '_')}.png")
    plt.close()

# =======================================================================
# K-MEANS CLUSTERING (UNSUPERVISED)
# =======================================================================

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(X_tfidf)

df["Cluster"] = clusters

# ----- PCA Scatter Plot -----
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf.toarray())

plt.figure(figsize=(10,7))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="tab10")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.title("K-Means Clustering (PCA 2D Visualization)", fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig("kmeans_pca_scatter.png")
plt.close()

# ----- Cluster Count Bar Chart -----
plt.figure(figsize=(8,5))
cluster_counts = df["Cluster"].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values, color=sns.color_palette("tab10"))
plt.title("K-Means Cluster Distribution", fontsize=14, weight="bold")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("kmeans_cluster_distribution.png")
plt.close()


# --------------------- ACCURACY BAR CHART ----------------------
model_names = [m[0] for m in metrics]
accuracies = [m[1] for m in metrics]
colors = ['#4E79A7', '#59A14F', '#F28E2B', '#9C51B6']

plt.figure(figsize=(9,6))
bars = plt.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=1.4)
plt.title("Model Accuracy Comparison (Premium Visualization)", fontsize=14, weight='bold')

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, h+0.01, f"{h:.2f}", ha='center', fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.close()

# --------------------- CATEGORY DISTRIBUTION ----------------------
plt.figure(figsize=(12,6))
df['Category'].value_counts().plot(kind='bar', color=sns.color_palette("Set3"), edgecolor='black')
plt.title("Category Distribution (Enhanced)", fontsize=14, weight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("category_distribution.png")
plt.close()

# --------------------- TF-IDF TOP WORDS ----------------------
idf_scores = tfidf.idf_
feature_names = np.array(tfidf.get_feature_names_out())
top_idx = np.argsort(idf_scores)[:20]
top_words = feature_names[top_idx]

colors2 = sns.color_palette("husl", len(top_words))

plt.figure(figsize=(10,7))
plt.barh(top_words, idf_scores[top_idx], color=colors2, edgecolor='black')
plt.gca().invert_yaxis()
plt.title("Top 20 TF-IDF Features (Enhanced Visualization)", fontsize=14, weight='bold')
plt.xlabel("TF-IDF Score")
plt.tight_layout()
plt.savefig("tfidf_top_words.png")
plt.close()

# =======================================================================
# BAR CHART (Accuracy vs Precision vs Recall)
# =======================================================================

models_list = [m[0] for m in metrics]
accuracy_list = [m[1] for m in metrics]
precision_list = [m[2] for m in metrics]
recall_list = [m[3] for m in metrics]

x = np.arange(len(models_list))
width = 0.25

plt.figure(figsize=(12,6))
plt.bar(x - width, accuracy_list, width, label='Accuracy', color='#4E79A7')
plt.bar(x, precision_list, width, label='Precision', color='#F28E2B')
plt.bar(x + width, recall_list, width, label='Recall', color='#59A14F')

plt.xlabel("Models", fontsize=12, weight='bold')
plt.ylabel("Scores", fontsize=12, weight='bold')
plt.title("Accuracy vs Precision vs Recall Comparison (All Models)", fontsize=14, weight='bold')

plt.xticks(x, models_list, rotation=25)
plt.legend()
plt.tight_layout()
plt.savefig("metrics_comparison_chart.png")
plt.close()

# =======================================================================
# SAVE METRICS CSV
# =======================================================================

metrics_df = pd.DataFrame(metrics, columns=["Model", "Accuracy", "Precision", "Recall"])
metrics_df.to_csv("model_metrics.csv", index=False)

print("\n====================================================")
print(" ALL TASKS DONE SUCCESSFULLY ")
print(" Generated Files:")
print(" - model_metrics.csv")
print(" - accuracy_plot.png")
print(" - category_distribution.png")
print(" - tfidf_top_words.png")
print(" - metrics_comparison_chart.png")
print(" - cm_* for all models")
print(" - kmeans_pca_scatter.png")
print(" - kmeans_cluster_distribution.png")
print("====================================================")
