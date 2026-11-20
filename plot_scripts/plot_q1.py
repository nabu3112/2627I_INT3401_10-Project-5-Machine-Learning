from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib.pyplot as plt
import backend
import models
import train

model = models.PerceptronModel(3)
dataset = backend.PerceptronDataset(model)
errors = train.train_perceptron(model, dataset)

plt.plot(range(1, len(errors) + 1), errors, marker="o")
plt.axhline(y=0, color="red", linestyle="--", label="Hội tụ: 0 lỗi")
plt.xlabel("Lượt quét dữ liệu")
plt.ylabel("Số mẫu phân loại sai")
plt.title("Q1: Số lỗi Perceptron theo lượt")
plt.grid(True)
plt.legend()
plt.tight_layout()

OUTPUT_DIR = PROJECT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / "q1_perceptron_errors.png"

plt.savefig(output_path, dpi=200)
print(f"Đã lưu biểu đồ tại: {output_path}")