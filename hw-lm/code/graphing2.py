import matplotlib.pyplot as plt

training_sizes = [1, 2, 4, 8]
error_rates = [12.22, 7.04, 6.67, 5.93]  

plt.plot(training_sizes, error_rates, marker='o')
plt.xlabel('Training Data Size (×original)')
plt.ylabel('Error Rate (%)')
plt.title('Learning Curve: Error Rate vs Training Data Size')
plt.grid(True)
plt.savefig('learning_curve.png', dpi=150)