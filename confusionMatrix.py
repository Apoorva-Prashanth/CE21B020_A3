"""import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Load CSV
df = pd.read_csv("C:/Users/APOORVA/Downloads/predictions_vanilla - test_predictions_embed256_hid128_enc2_dec1_LSTM_drop0.35_beam5.csv")

# Load custom Malayalam font
font_path = "C:/Users/APOORVA/Downloads/Noto_Sans_Malayalam/NotoSansMalayalam-VariableFont_wdth,wght.ttf"
mal_font = font_manager.FontProperties(fname=font_path)

# Extract characters
true_chars = []
pred_chars = []

for true, pred in zip(df["True Native"], df["Predicted Native"]):
    min_len = min(len(true), len(pred))
    for i in range(min_len):
        true_chars.append(true[i])
        pred_chars.append(pred[i])

# Create confusion matrix
labels = sorted(set(true_chars + pred_chars))
cm = confusion_matrix(true_chars, pred_chars, labels=labels)

# Plot confusion matrix manually (with Malayalam font)
fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(cm, cmap='Blues')

# Tick labels with Malayalam characters
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontproperties=mal_font, fontsize=10, rotation=90)
ax.set_yticklabels(labels, fontproperties=mal_font, fontsize=10)

# Title and color bar
ax.set_title("Character-Level Confusion Matrix (Malayalam)", fontproperties=mal_font, fontsize=16)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()"""
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Load CSV
df = pd.read_csv("C:/Users/APOORVA/Downloads/predictions_vanilla - test_predictions_embed256_hid128_enc2_dec1_LSTM_drop0.35_beam5.csv")

# Load custom Malayalam font
font_path = "C:/Users/APOORVA/Downloads/Noto_Sans_Malayalam/NotoSansMalayalam-VariableFont_wdth,wght.ttf"
mal_font = font_manager.FontProperties(fname=font_path)

# Extract characters
true_chars = []
pred_chars = []

for true, pred in zip(df["True Native"], df["Predicted Native"]):
    min_len = min(len(true), len(pred))
    for i in range(min_len):
        true_chars.append(true[i])
        pred_chars.append(pred[i])

# Unique sorted character labels
all_labels = sorted(set(true_chars + pred_chars))

# Batch size (adjustable)
batch_size = 30

# Plot confusion matrix for each batch
for i in range(0, len(all_labels), batch_size):
    batch_labels = all_labels[i:i+batch_size]
    
    # Filter out entries not in this batch
    filtered_true = [c for c, l in zip(true_chars, pred_chars) if c in batch_labels and l in batch_labels]
    filtered_pred = [l for c, l in zip(true_chars, pred_chars) if c in batch_labels and l in batch_labels]
    
    # Compute confusion matrix
    cm = confusion_matrix(filtered_true, filtered_pred, labels=batch_labels)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap='OrRd')

    ax.set_xticks(range(len(batch_labels)))
    ax.set_yticks(range(len(batch_labels)))
    ax.set_xticklabels(batch_labels, fontproperties=mal_font, fontsize=10, rotation=90)
    ax.set_yticklabels(batch_labels, fontproperties=mal_font, fontsize=10)

    ax.set_title(f"Confusion Matrix (Malayalam Characters {i+1}-{i+len(batch_labels)})", fontproperties=mal_font, fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

