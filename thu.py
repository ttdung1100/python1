import numpy as np  # optional
import pandas as pd
print(pd.__version__)
print(pd.show_versions(as_json=True))


ext install esbenp.prettier-vscode

import numpy as np  # optional


import numpy as np  # optional
import pandas as pd
print ("hello")


# Inputs
import numpy as np
mylist = list('abcedfghijklmnopqrstuvwxyz')
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))

# Solution
ser1 = pd.Series(mylist)
ser2 = pd.Series(myarr)
ser3 = pd.Series(mydict)
print(ser3.head())



# Input
import numpy as np
ser1 = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
ser2 = pd.Series(np.arange(26))

# Solution 1
df = pd.concat([ser1, ser2], axis=1)

# Solution 2
df = pd.DataFrame({'col1': ser1, 'col2': ser2})
print(df.head())



# Input
ser = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))

# Solution
ser.name = 'alphabets'
ser.head()



# Input
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])

# Solution
ser1[~ser1.isin(ser2)]


# Input
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])

# Solution
ser_u = pd.Series(np.union1d(ser1, ser2))  # union
ser_i = pd.Series(np.intersect1d(ser1, ser2))  # intersect
ser_u[~ser_u.isin(ser_i)]


# Input
state = np.random.RandomState(100)
ser = pd.Series(state.normal(10, 5, 25))

# Solution
np.percentile(ser, q=[0, 25, 50, 75, 100])



# Input
ser = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30)))

# Solution
ser.value_counts()



# Input
np.random.RandomState(100)
ser = pd.Series(np.random.randint(1, 5, [12]))

# Solution
print("Top 2 Freq:", ser.value_counts())
ser[~ser.isin(ser.value_counts().index[:2])] = 'Other'
ser



# Input
ser = pd.Series(np.random.random(20))
print(ser.head())

# Solution
pd.qcut(ser, q=[0, .10, .20, .3, .4, .5, .6, .7, .8, .9, 1], 
        labels=['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th']).head()
