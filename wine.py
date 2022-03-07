# 1. wine data download
#from urllib.request import urlretrieve
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/" + "winequality-white.csv"
#savepath = "winequality-white.csv"
#urlretrieve(url, savepath)  

# 2. wind data reading

#import pandas as pd
#df = pd.read_csv("winequality-white.csv", sep=";", encoding="utf-8")
#print(df)

# 3. wind 품질 판정하기 (초안)

from random import Random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


wine = pd.read_csv("winequality-white.csv", sep=";", encoding="utf-8")

#wine_g = wine.groupby(['quality'])["quality"].count()
#print(wine_g)
#wine_g.plot()
#plt.savefig("wine-count-plt.png")
#plt.show()

y = wine["quality"]
x = wine.drop("quality", axis=1)

#print(x)

newlist = []
for v in list(y) :
    if v <= 4 :
        newlist += [0]
    elif v <= 7 :
        newlist += [1]
    else :
        newlist += [2]

y = newlist
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))

print("정답률=", accuracy_score(y_test, y_pred))

# precision(정답률) : 정답과 예측 레이블 데이터 중에서 정답인 것의 비율
# recall(재현율) : 실제 정답인것 중에서 정답인것과 예측인것의 비율