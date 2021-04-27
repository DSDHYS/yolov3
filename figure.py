import matplotlib.pyplot as plt#约定俗成的写法plt
#首先定义两个函数（正弦&余弦）
import numpy as np

X=np.linspace(1,51,51)#-π to+π的256个值
c=[[8193.87163002]
 ,[7586.08943292]
 ,[7163.76215231]
 ,[6704.21145946]
 ,[6292.52500926]
 ,[5968.89410358]
 ,[5674.59433594]
 ,[5561.53731142]
 ,[5282.10787817]
 ,[5085.3803029 ]
 ,[4999.06942307]
 ,[4841.82463168]
 ,[4635.69729299]
 ,[4707.91224997]
 ,[4587.14318006]
 ,[4587.48528926]
 ,[4472.80101697]
 ,[4441.47905273]
 ,[4435.49333033]
 ,[4380.35937626]
 ,[4288.11156469]
 ,[4344.0401283 ]
 ,[4281.45436422]
 ,[4221.57817299]
 ,[4228.37809638]
 ,[4217.37942273]
 ,[4219.72267393]
 ,[4210.37378014]
 ,[4221.07755927]
 ,[4169.32504799]
 ,[4084.94670494]
 ,[4096.70771905]
 ,[4160.73313746]
 ,[4070.97845712]
 ,[4148.76288473]
 ,[4116.77212251]
 ,[4089.15582486]
 ,[4064.16594491]
 ,[4111.4169362 ]
 ,[4063.27800903]
 ,[4134.660493  ]
 ,[4051.89360646]
 ,[4029.41775828]
 ,[4050.33696331]
 ,[4055.20827721]
 ,[4088.26404272]
 ,[4059.74493366]
 ,[4045.65042767]
 ,[4066.00258621]
 ,[4075.494112  ]
 ,[4068.54159314]]
#c=np.array(c).reshape(1,51)
#在ipython的交互环境中需要这句话才能显示出来
plt.plot(X,c,label='c',color='g')
plt.show()