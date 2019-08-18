
import pandas as pd
import numpy as np
df=pd.read_csv('./q2/q2psnr.csv')
sum1=0
sum2=0
sum3=0
T91sigma2=open('./q2/T91sigma2.txt','w+')
T91sigma3=open('./q2/T91sigma3.txt','w+')
T91sigma4=open('./q2/T91sigma4.txt','w+')

for i in range(len(df['T91ratio=2'])):
    sum1+=df['T91ratio=2'][i]
    sum2 += df['T91ratio=3'][i]
    sum3 += df['T91ratio=4'][i]
    if (i+1) %14==0:
        T91sigma2.write(str(sum1/14.0)+'\n')
        T91sigma3.write(str(sum2/14.0)+'\n')
        T91sigma4.write(str(sum3/14.0)+'\n')
        sum1 = 0
        sum2 = 0
        sum3 = 0

T91sigma2.close()
T91sigma3.close()
T91sigma4.close()