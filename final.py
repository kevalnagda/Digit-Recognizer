import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('test.csv')
result = pd.read_csv('DR.csv')

data = data.values.astype('float32')
data = data.reshape(data.shape[0],28,28)

for i in range(0,25):
	plt.subplot(5,5,i+1)
	
	plt.imshow(data[i],cmap='gray')
	plt.title(result.iloc[i,1])
plt.show()

