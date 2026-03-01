import matplotlib.pyplot as plt
import re

log_file_path = '/home/auria/hackudc_lucas/inditex_dos/inditex/training.log'
output_plot_path = '/home/auria/hackudc_lucas/inditex_dos/inditex/loss_plot.png'

epochs = []
losses = []

with open(log_file_path, 'r') as f:
    for line in f:
        m = re.match(r'Epoch (\d+)/\d+ \| Loss: ([\d.]+)', line)
        if m:
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(2)))

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
plt.title('Domain Mapper Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Cosine Embedding Loss')
plt.grid(True)
plt.xticks(epochs)
plt.tight_layout()
plt.savefig(output_plot_path)
print(f"Plot saved to {output_plot_path}")