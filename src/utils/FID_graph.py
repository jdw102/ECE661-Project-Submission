import matplotlib.pyplot as plt

FID = [242.528671, 48.936172, 48.402069, 40.947289]
Labels = ["GAN", "WGAN", "AC-GAN", "DCGAN"]

bar_color = "#4472c4"

# Adjust the figure size
plt.figure(figsize=(8, 6))
plt.tight_layout()
plt.bar(Labels, FID, color=bar_color)
for i, v in enumerate(FID):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
plt.ylim(0, 270)
plt.xlabel("Model")
plt.ylabel("FID (Lower is Better)")
plt.title("FID Scores for Each Model")
plt.show()
