import pandas as pd

df = pd.read_csv("temper0.csv", encoding="utf-8")

md = {}

for i, row in df.iterrows() :

    m, d, v = (int(row['일시월']), int(row['일시일']), float(row['평균기온(°C)']))
    
    key = str(m) + "/" + str(d)
    if  not(key in md) :
        md[key] = []
    md[key] += [v]

#print(md)

avs = {}

for key in md :
    v = avs[key] = sum(md[key]) / len(md[key])
    #print("{0} : {1}".format(key, v))

print(avs["12/31"])



    #print("{0}, {1}, {2}".format(m, d, v))