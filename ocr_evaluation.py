import json
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def _ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def _save_training_curves(history, output_path, title):
    history_data = history.history
    epochs = range(1, len(history_data.get("accuracy", [])) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_data.get("accuracy", []), label="Train Accuracy")
    plt.plot(epochs, history_data.get("val_accuracy", []), label="Validation Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_data.get("loss", []), label="Train Loss")
    plt.plot(epochs, history_data.get("val_loss", []), label="Validation Loss")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def _save_confusion_matrix(y_true, y_pred, class_names, output_path, title):
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype(np.float32)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_normalized = cm_normalized / row_sums

    figure_size = max(12, len(class_names) * 0.28)
    plt.figure(figsize=(figure_size, figure_size))
    plt.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{title} - Confusion Matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_positions = np.arange(len(class_names))
    plt.xticks(tick_positions, class_names, rotation=90, fontsize=6)
    plt.yticks(tick_positions, class_names, fontsize=6)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = cm_normalized.max() * 0.5 if cm_normalized.size else 0
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            value = cm_normalized[i, j]
            if value > 0:
                plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=5,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def _save_prediction_samples(X_test, y_true_chars, y_pred_chars, output_path, title, max_samples=12):
    sample_count = min(max_samples, len(X_test))
    if sample_count == 0:
        return

    cols = 4
    rows = int(np.ceil(sample_count / cols))
    plt.figure(figsize=(cols * 3.2, rows * 3.2))

    for index in range(sample_count):
        ax = plt.subplot(rows, cols, index + 1)
        image = X_test[index].squeeze()
        ax.imshow(image, cmap="gray")
        ax.set_title(f"T:{y_true_chars[index]} | P:{y_pred_chars[index]}", fontsize=8)
        ax.axis("off")

    plt.suptitle(f"{title} - Sample Predictions", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def _update_benchmark_summary(summary_path, metrics_row):
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        summary_df = summary_df[summary_df["model_key"] != metrics_row["model_key"]]
        summary_df = pd.concat([summary_df, pd.DataFrame([metrics_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([metrics_row])

    summary_df = summary_df.sort_values("model_name").reset_index(drop=True)
    summary_df.to_csv(summary_path, index=False)
    return summary_df


def _save_benchmark_chart(summary_df, output_path):
    if len(summary_df) < 2:
        return

    models = summary_df["model_name"].tolist()
    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(13, 6))

    plt.subplot(1, 2, 1)
    plt.bar(x - width / 2, summary_df["strict_accuracy"], width, label="Strict Accuracy (%)")
    plt.bar(x + width / 2, summary_df["tolerant_accuracy"], width, label="Tolerant Accuracy (%)")
    plt.xticks(x, models, rotation=15)
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("OCR Accuracy Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(models, summary_df["avg_inference_time_ms"], color="#ff7f0e")
    plt.xticks(rotation=15)
    plt.ylabel("Average Inference Time (ms)")
    plt.title("OCR Speed Comparison")
    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def evaluate_ocr_model(
    model,
    X_test,
    y_test,
    label_encoder,
    output_dir,
    model_key,
    model_name,
    batch_size=32,
):
    _ensure_directory(output_dir)

    if len(X_test) == 0:
        raise ValueError("X_test is empty; cannot evaluate OCR model.")

    if len(X_test) > 0:
        _ = model.predict(X_test[:1], batch_size=1, verbose=0)

    start_time = time.perf_counter()
    predictions = model.predict(X_test, batch_size=batch_size, verbose=0)
    end_time = time.perf_counter()

    y_pred_encoded = np.argmax(predictions, axis=1)
    y_true_chars = label_encoder.inverse_transform(y_test)
    y_pred_chars = label_encoder.inverse_transform(y_pred_encoded)

    strict_correct = 0
    case_error_but_char_correct = 0
    total_wrong = 0

    for true_char, pred_char in zip(y_true_chars, y_pred_chars):
        if true_char == pred_char:
            strict_correct += 1
        elif true_char.lower() == pred_char.lower():
            case_error_but_char_correct += 1
        else:
            total_wrong += 1

    total_test = len(y_test)
    strict_accuracy = (strict_correct / total_test) * 100
    tolerant_accuracy = ((strict_correct + case_error_but_char_correct) / total_test) * 100
    total_inference_time = end_time - start_time
    average_time_per_image = (total_inference_time / total_test) * 1000

    class_names = list(label_encoder.classes_)

    classification_text = classification_report(
        y_test,
        y_pred_encoded,
        target_names=class_names,
        zero_division=0,
    )

    metrics_row = {
        "model_key": model_key,
        "model_name": model_name,
        "total_test": total_test,
        "strict_correct": strict_correct,
        "case_error_but_char_correct": case_error_but_char_correct,
        "total_wrong": total_wrong,
        "strict_accuracy": strict_accuracy,
        "tolerant_accuracy": tolerant_accuracy,
        "total_inference_time_sec": total_inference_time,
        "avg_inference_time_ms": average_time_per_image,
    }

    history_path = os.path.join(output_dir, f"training_curves_{model_key}.png")
    confusion_path = os.path.join(output_dir, f"confusion_matrix_{model_key}.png")
    samples_path = os.path.join(output_dir, f"prediction_samples_{model_key}.png")
    inference_path = os.path.join(output_dir, f"inference_time_{model_key}.png")
    report_path = os.path.join(output_dir, f"classification_report_{model_key}.txt")
    summary_path = os.path.join(output_dir, "ocr_benchmark_summary.csv")
    comparison_path = os.path.join(output_dir, "ocr_benchmark_comparison.png")

    summary_df = _update_benchmark_summary(summary_path, metrics_row)
    _save_benchmark_chart(summary_df, comparison_path)

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(f"Model: {model_name}\n")
        report_file.write(f"Total test samples: {total_test}\n\n")
        report_file.write(classification_text)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(["Total"], [total_inference_time], color="#1f77b4")
    plt.title("Total Inference Time")
    plt.ylabel("Seconds")
    plt.grid(axis="y", alpha=0.25)

    plt.subplot(1, 2, 2)
    plt.bar(["Average / Image"], [average_time_per_image], color="#ff7f0e")
    plt.title("Average Time per Image")
    plt.ylabel("Milliseconds")
    plt.grid(axis="y", alpha=0.25)

    plt.suptitle(f"{model_name} - Inference Time", y=1.02)
    plt.tight_layout()
    plt.savefig(inference_path, dpi=180, bbox_inches="tight")
    plt.close()

    return {
        "metrics": metrics_row,
        "history_path": history_path,
        "confusion_path": confusion_path,
        "samples_path": samples_path,
        "inference_path": inference_path,
        "report_path": report_path,
        "summary_path": summary_path,
        "comparison_path": comparison_path,
        "y_true_chars": y_true_chars,
        "y_pred_chars": y_pred_chars,
        "y_pred_encoded": y_pred_encoded,
        "prediction_probabilities": predictions,
    }


def save_ocr_evaluation_artifacts(history, X_test, y_test, label_encoder, model, output_dir, model_key, model_name, batch_size=32):
    _ensure_directory(output_dir)
    _save_training_curves(history, os.path.join(output_dir, f"training_curves_{model_key}.png"), model_name)

    results = evaluate_ocr_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder,
        output_dir=output_dir,
        model_key=model_key,
        model_name=model_name,
        batch_size=batch_size,
    )

    _save_confusion_matrix(
        y_true=y_test,
        y_pred=results["y_pred_encoded"],
        class_names=list(label_encoder.classes_),
        output_path=os.path.join(output_dir, f"confusion_matrix_{model_key}.png"),
        title=model_name,
    )
    _save_prediction_samples(
        X_test=X_test,
        y_true_chars=results["y_true_chars"],
        y_pred_chars=results["y_pred_chars"],
        output_path=os.path.join(output_dir, f"prediction_samples_{model_key}.png"),
        title=model_name,
    )

    return results