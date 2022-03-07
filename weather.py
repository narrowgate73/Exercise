# 1.  받은 기상테이터 가공
import pandas as pd

in_file = "OBS_ASOS_DD_20220306214142.csv"
out_file = "temper0.csv"

temper = pd.read_csv(in_file, sep=",", encoding="EUC-KR")
#print(temper)

df = pd.DataFrame(temper, columns=['일시', '평균기온(°C)'])

df['일시년'] = df.일시.str.split('-').str[0]
df['일시월'] = df.일시.str.split('-').str[1]
df['일시일'] = df.일시.str.split('-').str[2]

dg = pd.DataFrame(df, columns=['일시년', '일시월', '일시일', '평균기온(°C)'])

dg.to_csv(out_file, header=True, index=False)
#print(dg)