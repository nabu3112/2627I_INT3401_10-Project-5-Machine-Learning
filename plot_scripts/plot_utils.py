from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_DIR / "plots"
COUNTER_FILE = PLOTS_DIR / ".run_counter"


def start_run():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        number = int(COUNTER_FILE.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        number = 0

    number += 1
    COUNTER_FILE.write_text(str(number), encoding="utf-8")
    return number


def save_curve(question, run_number, values, title, ylabel,
               threshold=None, log_scale=False):
    if not values:
        return

    plt.figure(figsize=(8, 5))
    epochs = range(1, len(values) + 1)
    plt.plot(epochs, values, marker="o", markersize=3, label=ylabel)

    if threshold is not None:
        plt.axhline(
            threshold,
            color="black",
            linestyle="--",
            label=f"Ngưỡng ({threshold:g})"
        )

    if log_scale and all(value > 0 for value in values):
        plt.yscale("log")

    plt.xlabel("Epoch / lượt quét")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = PLOTS_DIR / f"{question}_run_{run_number:04d}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Đã lưu biểu đồ: {output_path}")


def save_attention(run_number, weights):
    if hasattr(weights, "detach"):
        weights = weights.detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    image = plt.imshow(weights, cmap="viridis", origin="lower", aspect="auto")
    plt.colorbar(image, label="Trọng số attention")
    plt.xlabel("Token được chú ý")
    plt.ylabel("Token truy vấn")
    plt.title("Q6: Ma trận attention")
    plt.tight_layout()

    output_path = PLOTS_DIR / f"q6_run_{run_number:04d}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Đã lưu biểu đồ: {output_path}")