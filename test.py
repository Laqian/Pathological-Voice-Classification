# 加载波士顿数据集
from sklearn.datasets import load_boston


data = load_boston()

X = data.data
#数据格式为 numpy.array
y = data.target

print(X)
print(y)
