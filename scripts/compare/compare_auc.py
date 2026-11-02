"""
So sánh AUC (%) và vẽ biểu đồ cột (Bar Chart) + đường cong ROC cho 4 thuật toán:
MIL (Sultani), RTFM, UR-DMU, VadCLIP.

Chạy lệnh từ thư mục HuanLuyen:
    python scripts/compare/compare_auc.py
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

GT_PATH = "scripts/vadclip/list/gt_ucf.npy"
OUT_DIR = "results/compare"
os.makedirs(OUT_DIR, exist_ok=True)

# Danh sách 4 thuật toán và các đường dẫn file điểm .npy tương ứng trong thư mục results/
MODEL_DIRS = [
    ("MIL (Sultani)", "results/mil_sultani"),
    ("RTFM",          "results/rtfm"),
    ("UR-DMU",        "results/urdmu"),
    ("VadCLIP",       "results/vadclip"),
]


def find_score_file(folder, gt_len):
    """Tự động tìm file frame_scores*.npy trong thư mục của từng thuật toán."""
    for name in ["frame_scores_best.npy", "frame_scores.npy", "scores.npy", "pred.npy"]:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p
    # Quét mọi file .npy trong folder có độ dài gần bằng gt_len
    for p in glob.glob(os.path.join(folder, "*.npy")):
        if "lens" not in os.path.basename(p):
            return p
    return None


def main():
    if not os.path.exists(GT_PATH):
        raise SystemExit(f"❌ Không tìm thấy file nhãn gốc tại: {GT_PATH}")

    gt = np.load(GT_PATH).ravel()
    rows = []
    rocs = {}

    for model_name, folder in MODEL_DIRS:
        score_path = find_score_file(folder, len(gt))
        if score_path is None:
            print(f"⚠️ [{model_name}] Chưa có file frame_scores.npy trong '{folder}' -> Bỏ qua.")
            continue

        scores = np.load(score_path).ravel()
        n = min(len(gt), len(scores))
        y_true, y_score = gt[:n], scores[:n]

        auc = roc_auc_score(y_true, y_score) * 100
        ap = average_precision_score(y_true, y_score) * 100
        fpr, tpr, _ = roc_curve(y_true, y_score)

        rows.append({
            "Thuật toán": model_name,
            "AUC (%)": round(auc, 2),
            "AP (%)": round(ap, 2),
            "Số frame": len(scores),
            "File nguồn": score_path.replace("\\", "/")
        })
        rocs[model_name] = (fpr, tpr, auc)

    if not rows:
        raise SystemExit("❌ Không tìm thấy bất kỳ file frame_scores.npy nào trong results/!")

    # 1. In bảng kết quả và xuất ra CSV
    df = pd.DataFrame(rows)
    print("\n================ BẢNG SO SÁNH AUC 4 THUẬT TOÁN ================")
    print(df.to_string(index=False))
    print("===============================================================")

    csv_path = os.path.join(OUT_DIR, "auc_comparison.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Đã lưu bảng số liệu CSV: {csv_path}")

    # 2. Vẽ biểu đồ cột (Bar Chart) so sánh AUC
    plt.figure(figsize=(8.5, 5.2), dpi=150)
    palette = ["#5c92c8", "#2b7bba", "#08519c", "#d95f02"]
    bars = plt.bar(df["Thuật toán"], df["AUC (%)"], color=palette[:len(df)], width=0.48, edgecolor="black", zorder=3)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.35, f"{h:.2f}%",
                 ha="center", va="bottom", fontweight="bold", fontsize=11)

    min_auc = max(50, min(df["AUC (%)"]) - 8)
    max_auc = min(100, max(df["AUC (%)"]) + 5)
    plt.ylim(min_auc, max_auc)
    plt.ylabel("Frame-level AUC (%)", fontsize=12, fontweight="bold")
    plt.xlabel("Thuật toán", fontsize=12, fontweight="bold")
    plt.title("So sánh AUC (%) giữa các thuật toán trên tập UCF-Crime", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    plt.tight_layout()

    bar_path = os.path.join(OUT_DIR, "auc_bar_chart.png")
    plt.savefig(bar_path)
    plt.close()
    print(f"✅ Đã lưu biểu đồ cột AUC: {bar_path}")

    # 3. Vẽ thêm đường cong ROC của cả 4 thuật toán trên cùng 1 hình
    plt.figure(figsize=(7, 6), dpi=150)
    for (model_name, (fpr, tpr, auc)), color in zip(rocs.items(), palette):
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f}%)", linewidth=2, color=color)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Ngẫu nhiên (50.00%)")
    plt.xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (TPR)", fontsize=11, fontweight="bold")
    plt.title("Đường cong ROC trên tập UCF-Crime", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid( linestyle="--", alpha=0.5)
    plt.tight_layout()

    roc_path = os.path.join(OUT_DIR, "roc_curve_4models.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"✅ Đã lưu biểu đồ đường cong ROC: {roc_path}")


if __name__ == "__main__":
    main()