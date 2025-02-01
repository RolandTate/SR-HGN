import matplotlib.pyplot as plt

# Data for hidden_dim
hidden_dim_values = [8, 16, 32, 64, 128, 256, 512]
acm_hidden_dim = [95.54, 97.15, 95.82, 95.62, 94.68, 94.18, 91.93]
dblp_hidden_dim = [96.12, 96.93, 95.70, 96.24, 96.22, 95.87, 93.83]

# Data for lambda
lambda_values = [0.1, 0.2, 0.5, 1, 2, 3, 5]
acm_lambda = [97.15, 95.62, 95.45, 95.42, 94.60, 94.48, 94.63]
dblp_lambda = [96.93, 96.07, 96.24, 96.81, 96.36, 96.76, 96.58]

# Data for N of Layers
layers_values = [1, 2, 3, 4, 5, 6, 7]
acm_layers = [94.53, 95.67, 97.15, 95.30, 95.12, 94.55, 92.92]
dblp_layers = [92.33, 95.36, 96.93, 96.00, 96.88, 96.22, 96.51]

# Create separate plots for each variable
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
# Plot for hidden_dim with equally spaced x-axis
plt.figure(figsize=(8, 6))
plt.plot(range(len(hidden_dim_values)), acm_hidden_dim, marker='o', label='ACM', linewidth=2)
plt.plot(range(len(hidden_dim_values)), dblp_hidden_dim, marker='o', label='DBLP', linewidth=2)
# plt.xlabel('Hidden Dimension', fontsize=28)
plt.xticks(range(len(hidden_dim_values)), hidden_dim_values, fontsize=24, fontname='Times New Roman')
plt.xlabel(r'$d_h$', fontsize=28)
plt.ylabel('Micro-F1', fontsize=28, fontname='Times New Roman')
plt.ylim(90, 100)
plt.yticks(fontsize=24, fontname='Times New Roman')
plt.legend(fontsize=24, prop={'family': 'Times New Roman', 'size': 24}, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2)
plt.grid(False)
plt.tight_layout()
plt.show()

# Plot for lambda with equally spaced x-axis
plt.figure(figsize=(8, 6))
plt.plot(range(len(lambda_values)), acm_lambda, marker='o', label='ACM', linewidth=2)
plt.plot(range(len(lambda_values)), dblp_lambda, marker='o', label='DBLP', linewidth=2)
# plt.xlabel(r'$\lambda$', fontsize=28)
plt.xticks(range(len(lambda_values)), lambda_values, fontsize=24, fontname='Times New Roman')
plt.xlabel(r'$\lambda$', fontsize=28)
plt.ylabel('Micro-F1', fontsize=28, fontname='Times New Roman')
plt.ylim(90, 100)
plt.yticks(fontsize=24, fontname='Times New Roman')
plt.legend(fontsize=24, prop={'family': 'Times New Roman', 'size': 24}, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2)
plt.grid(False)
plt.tight_layout()
plt.show()

# Plot for N of Layers (no change needed as this is already equally spaced)
plt.figure(figsize=(8, 6))
plt.plot(layers_values, acm_layers, marker='o', label='ACM', linewidth=2)
plt.plot(layers_values, dblp_layers, marker='o', label='DBLP', linewidth=2)
# plt.xlabel('Number of Layers', fontsize=28)
plt.xticks(layers_values, fontsize=24, fontname='Times New Roman')
plt.xlabel(r'$L$', fontsize=28)
plt.ylabel('Micro-F1', fontsize=28, fontname='Times New Roman')
plt.ylim(90, 100)
plt.yticks(fontsize=24, fontname='Times New Roman')
plt.legend(fontsize=24, prop={'family': 'Times New Roman', 'size': 24}, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2)
plt.grid(False)
plt.tight_layout()
plt.show()


# Data for hidden_dim (NMI)
hidden_dim_values = [8, 16, 32, 64, 128, 256, 512]
acm_hidden_dim_nmi = [90.27, 92.27, 88.04, 83.71, 85.65, 82.49, 72.22]
dblp_hidden_dim_nmi = [91.69, 91.33, 90.86, 87.96, 90.57, 88.44, 79.96]

# Data for lambda (NMI)
lambda_values = [0.1, 0.2, 0.5, 1, 2, 3, 5]
acm_lambda_nmi = [92.27, 92.02, 86.71, 88.89, 84.38, 80.67, 81.88]
dblp_lambda_nmi = [91.33, 91.09, 90.42, 89.87, 88.87, 89.17, 87.45]

# Data for N of Layers (NMI)
layers_values = [1, 2, 3, 4, 5, 6, 7]
acm_layers_nmi = [86.74, 87.20, 92.27, 91.79, 84.72, 82.74, 78.18]
dblp_layers_nmi = [91.15, 92.09, 91.33, 90.87, 87.65, 83.28, 82.96]

# Create separate plots for each variable

# Plot for hidden_dim (NMI) with equally spaced x-axis
plt.figure(figsize=(8, 6))
plt.plot(range(len(hidden_dim_values)), acm_hidden_dim_nmi, marker='o', label='ACM', linewidth=2)
plt.plot(range(len(hidden_dim_values)), dblp_hidden_dim_nmi, marker='o', label='DBLP', linewidth=2)
# plt.xlabel('Hidden Dimension', fontsize=28)
plt.xticks(range(len(hidden_dim_values)), hidden_dim_values, fontsize=24, fontname='Times New Roman')
plt.xlabel(r'$d_h$', fontsize=28)
plt.ylabel('NMI', fontsize=28, fontname='Times New Roman')
plt.ylim(70, 100)
plt.yticks(fontsize=24, fontname='Times New Roman')
plt.legend(fontsize=24, prop={'family': 'Times New Roman', 'size': 24}, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2)
plt.grid(False)
plt.tight_layout()
plt.show()

# Plot for lambda (NMI) with equally spaced x-axis, using λ symbol
plt.figure(figsize=(8, 6))
plt.plot(range(len(lambda_values)), acm_lambda_nmi, marker='o', label='ACM', linewidth=2)
plt.plot(range(len(lambda_values)), dblp_lambda_nmi, marker='o', label='DBLP', linewidth=2)
# plt.xlabel(r'$\lambda$', fontsize=28)
plt.xticks(range(len(lambda_values)), lambda_values, fontsize=24, fontname='Times New Roman')
plt.xlabel(r'$\lambda$', fontsize=28)
plt.ylabel('NMI', fontsize=28, fontname='Times New Roman')
plt.ylim(70, 100)
plt.yticks(fontsize=24, fontname='Times New Roman')
plt.legend(fontsize=24, prop={'family': 'Times New Roman', 'size': 24}, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2)
plt.grid(False)
plt.tight_layout()
plt.show()

# Plot for N of Layers (NMI) with equally spaced x-axis
plt.figure(figsize=(8, 6))
plt.plot(layers_values, acm_layers_nmi, marker='o', label='ACM', linewidth=2)
plt.plot(layers_values, dblp_layers_nmi, marker='o', label='DBLP', linewidth=2)
#plt.xlabel('Number of Layers', fontsize=28)
plt.xticks(layers_values, fontsize=24, fontname='Times New Roman')
plt.xlabel(r'$L$', fontsize=28)
plt.ylabel('NMI', fontsize=28, fontname='Times New Roman')
plt.ylim(70, 100)
plt.yticks(fontsize=24, fontname='Times New Roman')
plt.legend(fontsize=24, prop={'family': 'Times New Roman', 'size': 24}, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2)
plt.grid(False)
plt.tight_layout()
plt.show()
