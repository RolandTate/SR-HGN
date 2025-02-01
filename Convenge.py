from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

# Initialize parameters
params = {'epochs': 200}

# Generate random data for demonstration purposes
train_losses = np.exp(-np.linspace(0, 5, params['epochs'])) + 0.1 * np.random.rand(params['epochs'])
train_micro_values = np.linspace(0.2, 1.0, params['epochs']) + 0.05 * np.random.rand(params['epochs'])
val_micro_values = np.linspace(0.3, 0.9, params['epochs']) + 0.05 * np.random.rand(params['epochs'])

fs = 28
ls = 24
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = fs
fig, ax1 = plt.subplots(figsize=(10, 8))

ax1.set_xlabel('Epochs', fontsize=fs)
ax1.set_ylabel('Loss Values', fontsize=fs, labelpad=10)

# Set limits to start from 0 and align spines
ax1.set_xlim(0, params['epochs'])
ax1.set_ylim(0, max(train_losses)+ 0.02*max(train_losses))
ax1.spines['left'].set_position('zero')
ax1.spines['bottom'].set_position('zero')

train_loss_line, = ax1.plot(range(params['epochs']), train_losses, color='#1A2A3A', label='Loss')
ax1.tick_params(axis='y', labelsize=ls)
ax1.tick_params(axis='x', labelsize=ls)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Micro-F1', fontsize=fs, labelpad=10)  # we already handled the x-label with ax1
ax2.set_ylim(0, 1.02)
ax2.spines['right'].set_position(('outward', 0))
micro_line, = ax2.plot(range(params['epochs']), train_micro_values, color='#F25022', label='Training')
macro_line, = ax2.plot(range(params['epochs']), val_micro_values, color='#FFB900', label='Validation')
ax2.tick_params(axis='y', labelsize=ls)

lines = [train_loss_line, micro_line, macro_line]
labels = [line.get_label() for line in lines]

# Create the legend manually
plt.subplots_adjust(top=0.9)  # Leave space at the top for the legend

# Create the legend manually and position it in the remaining top space
plt.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fontsize=ls+1)

plt.show()