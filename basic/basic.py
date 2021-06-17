import numpy as np

a = np.arange(6)
a2 = a[np.newaxis, :]
a3 = np.reshape(a, (2, 3), 'C')

print(a2.shape)
print(a3)