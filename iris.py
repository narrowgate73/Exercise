import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# iris 데이터를 읽어들인다
iris_data = pd.read_csv('iris.csv', encoding='utf-8')

# iris 데이터를 레이블과 입력데이터로 분리
y = iris_data.loc[:, "variety"]
x = iris_data.loc[:, ["sepal.length", "sepal.width", "petal.length", "petal.width"]]

# 학습전용과 테스트전용 데이터 분리

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)


# 학습하기

clf = SVC()
clf.fit(x_train, y_train)

# 평가하기

y_pred = clf.predict(x_test)
print('정답률=', accuracy_score(y_test, y_pred))


#print(x_train)
#print(x_test)
