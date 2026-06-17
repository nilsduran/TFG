import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Set styling for a clean, professional thesis look
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)

# Generate Gamma distribution data (typical parameters for binocular rivalry: shape alpha=3-4)
a = 3.5
x = np.linspace(0, 10, 1000)
y = gamma.pdf(x, a, scale=1.0)

# Plot the distribution
ax.plot(x, y, color='#1f77b4', linewidth=2.5, label='Gamma Distribution')
ax.fill_between(x, y, color='#1f77b4', alpha=0.15)

# Style axes
ax.set_title('Typical Perceptual Dominance Durations', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Dominance Duration (seconds)', fontsize=10)
ax.set_ylabel('Probability Density', fontsize=10)
ax.set_xlim(0, 10)
ax.set_ylim(0, max(y) * 1.1)

# Clean up spines and grids
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.5)

# Add a vertical line for the mean or median to illustrate skewness
mean_val = a * 1.0
ax.axvline(mean_val, color='#d62728', linestyle=':', linewidth=2, label=f'Mean Duration ({mean_val}s)')

ax.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()

# Save the plot
plt.savefig('docs/images/gamma.png', bbox_inches='tight')
print("Successfully generated docs/images/gamma.png")
