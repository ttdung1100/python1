{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "source": [
    "import numpy as np  # optional\r\n",
    "import pandas as pd\r\n",
    "print(pd.__version__)\r\n",
    "print(pd.show_versions(as_json=True))"
   ],
   "outputs": [
    {
     "output_type": "error",
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'numpy'",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32mC:\\Users\\DANGDU~1\\AppData\\Local\\Temp/ipykernel_8996/823475755.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[1;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mnp\u001b[0m  \u001b[1;31m# optional\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mpandas\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__version__\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshow_versions\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mas_json\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'numpy'"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "ext install esbenp.prettier-vscode"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "import numpy as np  # optional"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "import numpy as np  # optional\r\n",
    "import pandas as pd\r\n",
    "print (\"hello\")"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Inputs\r\n",
    "import numpy as np\r\n",
    "mylist = list('abcedfghijklmnopqrstuvwxyz')\r\n",
    "myarr = np.arange(26)\r\n",
    "mydict = dict(zip(mylist, myarr))\r\n",
    "\r\n",
    "# Solution\r\n",
    "ser1 = pd.Series(mylist)\r\n",
    "ser2 = pd.Series(myarr)\r\n",
    "ser3 = pd.Series(mydict)\r\n",
    "print(ser3.head())\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "mylist = list('abcedfghijklmnopqrstuvwxyz')\r\n",
    "myarr = np.arange(26)\r\n",
    "mydict = dict(zip(mylist, myarr))\r\n",
    "ser = pd.Series(mydict)\r\n",
    "\r\n",
    "# Solution\r\n",
    "df = ser.to_frame().reset_index()\r\n",
    "print(df.head())\r\n"
   ],
   "outputs": [
    {
     "output_type": "error",
     "ename": "NameError",
     "evalue": "name 'np' is not defined",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32mC:\\Users\\DANGDU~1\\AppData\\Local\\Temp/ipykernel_8996/3419209086.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# Input\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[0mmylist\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlist\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'abcedfghijklmnopqrstuvwxyz'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[0mmyarr\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0marange\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m26\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[0mmydict\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mdict\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mzip\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmylist\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmyarr\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mser\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mSeries\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmydict\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'np' is not defined"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "import numpy as np\r\n",
    "ser1 = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))\r\n",
    "ser2 = pd.Series(np.arange(26))\r\n",
    "\r\n",
    "# Solution 1\r\n",
    "df = pd.concat([ser1, ser2], axis=1)\r\n",
    "\r\n",
    "# Solution 2\r\n",
    "df = pd.DataFrame({'col1': ser1, 'col2': ser2})\r\n",
    "print(df.head())\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))\r\n",
    "\r\n",
    "# Solution\r\n",
    "ser.name = 'alphabets'\r\n",
    "ser.head()\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser1 = pd.Series([1, 2, 3, 4, 5])\r\n",
    "ser2 = pd.Series([4, 5, 6, 7, 8])\r\n",
    "\r\n",
    "# Solution\r\n",
    "ser1[~ser1.isin(ser2)]\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser1 = pd.Series([1, 2, 3, 4, 5])\r\n",
    "ser2 = pd.Series([4, 5, 6, 7, 8])\r\n",
    "\r\n",
    "# Solution\r\n",
    "ser_u = pd.Series(np.union1d(ser1, ser2))  # union\r\n",
    "ser_i = pd.Series(np.intersect1d(ser1, ser2))  # intersect\r\n",
    "ser_u[~ser_u.isin(ser_i)]\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "state = np.random.RandomState(100)\r\n",
    "ser = pd.Series(state.normal(10, 5, 25))\r\n",
    "\r\n",
    "# Solution\r\n",
    "np.percentile(ser, q=[0, 25, 50, 75, 100])\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30)))\r\n",
    "\r\n",
    "# Solution\r\n",
    "ser.value_counts()\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "np.random.RandomState(100)\r\n",
    "ser = pd.Series(np.random.randint(1, 5, [12]))\r\n",
    "\r\n",
    "# Solution\r\n",
    "print(\"Top 2 Freq:\", ser.value_counts())\r\n",
    "ser[~ser.isin(ser.value_counts().index[:2])] = 'Other'\r\n",
    "ser\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser = pd.Series(np.random.random(20))\r\n",
    "print(ser.head())\r\n",
    "\r\n",
    "# Solution\r\n",
    "pd.qcut(ser, q=[0, .10, .20, .3, .4, .5, .6, .7, .8, .9, 1], \r\n",
    "        labels=['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th']).head()\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser = pd.Series(np.random.randint(1, 10, 35))\r\n",
    "\r\n",
    "# Solution\r\n",
    "df = pd.DataFrame(ser.values.reshape(7,5))\r\n",
    "print(df)\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser = pd.Series(np.random.randint(1, 10, 7))\r\n",
    "ser\r\n",
    "\r\n",
    "# Solution\r\n",
    "print(ser)\r\n",
    "np.argwhere(ser % 3==0)\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))\r\n",
    "pos = [0, 4, 8, 14, 20]\r\n",
    "\r\n",
    "# Solution\r\n",
    "ser.take(pos)\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser1 = pd.Series(range(5))\r\n",
    "ser2 = pd.Series(list('abcde'))\r\n",
    "\r\n",
    "# Output\r\n",
    "# Vertical\r\n",
    "ser1.append(ser2)\r\n",
    "\r\n",
    "# Horizontal\r\n",
    "df = pd.concat([ser1, ser2], axis=1)\r\n",
    "print(df)\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser1 = pd.Series([10, 9, 6, 5, 3, 1, 12, 8, 13])\r\n",
    "ser2 = pd.Series([1, 3, 10, 13])\r\n",
    "\r\n",
    "# Solution 1\r\n",
    "[np.where(i == ser1)[0].tolist()[0] for i in ser2]\r\n",
    "\r\n",
    "# Solution 2\r\n",
    "[pd.Index(ser1).get_loc(i) for i in ser2]\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "truth = pd.Series(range(10))\r\n",
    "pred = pd.Series(range(10)) + np.random.random(10)\r\n",
    "\r\n",
    "# Solution\r\n",
    "np.mean((truth-pred)**2)\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser = pd.Series(['how', 'to', 'kick', 'ass?'])\r\n",
    "\r\n",
    "# Solution 1\r\n",
    "ser.map(lambda x: x.title())\r\n",
    "\r\n",
    "# Solution 2\r\n",
    "ser.map(lambda x: x[0].upper() + x[1:])\r\n",
    "\r\n",
    "# Solution 3\r\n",
    "pd.Series([i.title() for i in ser])\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser = pd.Series(['how', 'to', 'kick', 'ass?'])\r\n",
    "\r\n",
    "# Solution\r\n",
    "ser.map(lambda x: len(x))\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser = pd.Series([1, 3, 6, 10, 15, 21, 27, 35])\r\n",
    "\r\n",
    "# Solution\r\n",
    "print(ser.diff().tolist())\r\n",
    "print(ser.diff().diff().tolist())\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])\r\n",
    "\r\n",
    "# Solution 1\r\n",
    "from dateutil.parser import parse\r\n",
    "ser.map(lambda x: parse(x))\r\n",
    "\r\n",
    "# Solution 2\r\n",
    "pd.to_datetime(ser)\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])\r\n",
    "\r\n",
    "# Solution\r\n",
    "from dateutil.parser import parse\r\n",
    "ser_ts = ser.map(lambda x: parse(x))\r\n",
    "\r\n",
    "# day of month\r\n",
    "print(\"Date: \", ser_ts.dt.day.tolist())\r\n",
    "\r\n",
    "# week number\r\n",
    "print(\"Week number: \", ser_ts.dt.weekofyear.tolist())\r\n",
    "\r\n",
    "# day of year\r\n",
    "print(\"Day number of year: \", ser_ts.dt.dayofyear.tolist())\r\n",
    "\r\n",
    "# day of week\r\n",
    "print(\"Day of week: \", ser_ts.dt.weekday_name.tolist())\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "import pandas as pd\r\n",
    "# Input\r\n",
    "ser = pd.Series(['Jan 2010', 'Feb 2011', 'Mar 2012'])\r\n",
    "\r\n",
    "# Solution 1\r\n",
    "from dateutil.parser import parse\r\n",
    "# Parse the date\r\n",
    "ser_ts = ser.map(lambda x: parse(x))\r\n",
    "\r\n",
    "# Construct date string with date as 4\r\n",
    "ser_datestr = ser_ts.dt.year.astype('str') + '-' + ser_ts.dt.month.astype('str') + '-' + '04'\r\n",
    "\r\n",
    "# Format it.\r\n",
    "[parse(i).strftime('%Y-%m-%d') for i in ser_datestr]\r\n",
    "\r\n",
    "# Solution 2\r\n",
    "ser.map(lambda x: parse('04 ' + x))\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "ser = pd.Series(['Apple', 'Orange', 'Plan', 'Python', 'Money'])\r\n",
    "\r\n",
    "# Solution\r\n",
    "from collections import Counter\r\n",
    "mask = ser.map(lambda x: sum([Counter(x.lower()).get(i, 0) for i in list('aeiou')]) >= 2)\r\n",
    "ser[mask]\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "emails = pd.Series(['buying books at amazom.com', 'rameses@egypt.com', 'matt@t.co', 'narendra@modi.com'])\n",
    "\n",
    "# Solution 1 (as series of strings)\n",
    "import re\n",
    "pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\\\.[A-Za-z]{2,4}'\n",
    "mask = emails.map(lambda x: bool(re.match(pattern, x)))\n",
    "emails[mask]\n",
    "\n",
    "# Solution 2 (as series of list)\n",
    "emails.str.findall(pattern, flags=re.IGNORECASE)\n",
    "\n",
    "# Solution 3 (as list)\n",
    "[x[0] for x in [re.findall(pattern, email) for email in emails] if len(x) > 0]\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "fruit = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))\n",
    "weights = pd.Series(np.linspace(1, 10, 10))\n",
    "\n",
    "# Solution\n",
    "weights.groupby(fruit).mean()\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "p = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])\n",
    "q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])\n",
    "\n",
    "# Solution \n",
    "sum((p - q)**2)**.5\n",
    "\n",
    "# Solution (using func)\n",
    "np.linalg.norm(p-q)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])\n",
    "\n",
    "# Solution\n",
    "dd = np.diff(np.sign(np.diff(ser)))\n",
    "peak_locs = np.where(dd == -2)[0] + 1\n",
    "peak_locs\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "my_str = 'dbc deb abed gade'\n",
    "\n",
    "# Solution\n",
    "ser = pd.Series(list('dbc deb abed gade'))\n",
    "freq = ser.value_counts()\n",
    "print(freq)\n",
    "least_freq = freq.dropna().index[-1]\n",
    "\"\".join(ser.replace(' ', least_freq))\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Solution\n",
    "ser = pd.Series(np.random.randint(1,10,10), pd.date_range('2000-01-01', periods=10, freq='W-SAT'))\n",
    "ser\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "ser = pd.Series([1,10,3, np.nan], index=pd.to_datetime(['2000-01-01', '2000-01-03', '2000-01-06', '2000-01-08']))\n",
    "\n",
    "# Solution\n",
    "ser.resample('D').ffill()  # fill with previous value\n",
    "\n",
    "# Alternatives\n",
    "ser.resample('D').bfill()  # fill with next value\n",
    "ser.resample('D').bfill().ffill()  # fill next else prev value\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "ser = pd.Series(np.arange(20) + np.random.normal(1, 10, 20))\n",
    "\n",
    "# Solution\n",
    "autocorrelations = [ser.autocorr(i).round(2) for i in range(11)]\n",
    "print(autocorrelations[1:])\n",
    "print('Lag having highest correlation: ', np.argmax(np.abs(autocorrelations[1:]))+1)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Solution 1: Use chunks and for-loop\n",
    "df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', chunksize=50)\n",
    "df2 = pd.DataFrame()\n",
    "for chunk in df:\n",
    "    df2 = df2.append(chunk.iloc[0,:])"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Solution 2: Use chunks and list comprehension\n",
    "df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', chunksize=50)\n",
    "df2 = pd.concat([chunk.iloc[0] for chunk in df], axis=1)\n",
    "df2 = df2.transpose()"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Solution 3: Use csv reader\n",
    "import csv          \n",
    "with open('BostonHousing.csv', 'r') as f:\n",
    "    reader = csv.reader(f)\n",
    "    out = []\n",
    "    for i, row in enumerate(reader):\n",
    "        if i%50 == 0:\n",
    "            out.append(row)\n",
    "            \n",
    "df2 = pd.DataFrame(out[1:], columns=out[0])\n",
    "print(df2.head())"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "L = pd.Series(range(15))\n",
    "\n",
    "def gen_strides(a, stride_len=5, window_len=5):\n",
    "    n_strides = ((a.size-window_len)//stride_len) + 1\n",
    "    return np.array([a[s:(s+window_len)] for s in np.arange(0, a.size, stride_len)[:n_strides]])\n",
    "\n",
    "gen_strides(L, stride_len=2, window_len=4)"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', usecols=['crim', 'medv'])\n",
    "print(df.head())"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')\n",
    "\n",
    "#  number of rows and columns\n",
    "print(df.shape)\n",
    "\n",
    "# datatypes\n",
    "print(df.dtypes)\n",
    "\n",
    "# how many columns under each dtype\n",
    "print(df.get_dtype_counts())\n",
    "print(df.dtypes.value_counts())\n",
    "\n",
    "# summary statistics\n",
    "df_stats = df.describe()\n",
    "\n",
    "# numpy array \n",
    "df_arr = df.values\n",
    "\n",
    "# list\n",
    "df_list = df.values.tolist()"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')\n",
    "\n",
    "# Solution\n",
    "# Get Manufacturer with highest price\n",
    "df.loc[df.Price == np.max(df.Price), ['Manufacturer', 'Model', 'Type']]\n",
    "\n",
    "# Get Row and Column number\n",
    "row, col = np.where(df.values == np.max(df.Price))\n",
    "\n",
    "# Get the value\n",
    "df.iat[row[0], col[0]]\n",
    "df.iloc[row[0], col[0]]\n",
    "\n",
    "# Alternates\n",
    "df.at[row[0], 'Price']\n",
    "df.get_value(row[0], 'Price')\n",
    "\n",
    "# The difference between `iat` - `iloc` vs `at` - `loc` is:\n",
    "# `iat` snd `iloc` accepts row and column numbers. \n",
    "# Whereas `at` and `loc` accepts index and column names.\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')\n",
    "\n",
    "# Solution\n",
    "# Step 1:\n",
    "df=df.rename(columns = {'Type':'CarType'})\n",
    "# or\n",
    "df.columns.values[2] = \"CarType\"\n",
    "\n",
    "# Step 2:\n",
    "df.columns = df.columns.map(lambda x: x.replace('.', '_'))\n",
    "print(df.columns)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')\n",
    "\n",
    "# Solution\n",
    "df.isnull().values.any()\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')\n",
    "\n",
    "# Solution\n",
    "n_missings_each_col = df.apply(lambda x: x.isnull().sum())\n",
    "n_missings_each_col.argmax()\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')\n",
    "\n",
    "# Solution\n",
    "df_out = df[['Min.Price', 'Max.Price']] = df[['Min.Price', 'Max.Price']].apply(lambda x: x.fillna(x.mean()))\n",
    "print(df_out.head())\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')\n",
    "\n",
    "# Solution\n",
    "d = {'Min.Price': np.nanmean, 'Max.Price': np.nanmedian}\n",
    "df[['Min.Price', 'Max.Price']] = df[['Min.Price', 'Max.Price']].apply(lambda x, d: x.fillna(d[x.name](x)), args=(d, ))\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))\n",
    "\n",
    "# Solution\n",
    "type(df[['a']])\n",
    "type(df.loc[:, ['a']])\n",
    "type(df.iloc[:, [0]])\n",
    "\n",
    "# Alternately the following returns a Series\n",
    "type(df.a)\n",
    "type(df['a'])\n",
    "type(df.loc[:, 'a'])\n",
    "type(df.iloc[:, 1])\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))\n",
    "\n",
    "# Solution Q1\n",
    "df[list('cbade')]\n",
    "\n",
    "# Solution Q2 - No hard coding\n",
    "def switch_columns(df, col1=None, col2=None):\n",
    "    colnames = df.columns.tolist()\n",
    "    i1, i2 = colnames.index(col1), colnames.index(col2)\n",
    "    colnames[i2], colnames[i1] = colnames[i1], colnames[i2]\n",
    "    return df[colnames]\n",
    "\n",
    "df1 = switch_columns(df, 'a', 'c')\n",
    "\n",
    "# Solution Q3\n",
    "df[sorted(df.columns)]\n",
    "# or\n",
    "df.sort_index(axis=1, ascending=False, inplace=True)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')\n",
    "\n",
    "# Solution\n",
    "pd.set_option('display.max_columns', 10)\n",
    "pd.set_option('display.max_rows', 10)\n",
    "# df\n",
    "\n",
    "# Show all available options\n",
    "# pd.describe_option()\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.random.random(4)**10, columns=['random'])\n",
    "\n",
    "# Solution 1: Rounding\n",
    "df.round(4)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Solution 2: Use apply to change format\n",
    "df.apply(lambda x: '%.4f' % x, axis=1)\n",
    "# or\n",
    "df.applymap(lambda x: '%.4f' % x)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Solution 3: Use set_option\n",
    "pd.set_option('display.float_format', lambda x: '%.4f' % x)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Solution 4: Assign display.float_format\n",
    "pd.options.display.float_format = '{:.4f}'.format\n",
    "print(df)\n",
    "# Reset/undo float formatting\n",
    "pd.options.display.float_format = None\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.random.random(4), columns=['random'])\n",
    "\n",
    "# Solution\n",
    "out = df.style.format({\n",
    "    'random': '{0:.2%}'.format,\n",
    "})\n",
    "\n",
    "out\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')\n",
    "\n",
    "# Solution\n",
    "print(df.iloc[::20, :][['Manufacturer', 'Model', 'Type']])\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv', usecols=[0,1,2,3,5])\n",
    "\n",
    "# Solution\n",
    "df[['Manufacturer', 'Model', 'Type']] = df[['Manufacturer', 'Model', 'Type']].fillna('missing')\n",
    "df.index = df.Manufacturer + '_' + df.Model + '_' + df.Type\n",
    "print(df.index.is_unique)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.random.randint(1, 30, 30).reshape(10,-1), columns=list('abc'))\n",
    "\n",
    "# Solution\n",
    "n = 5\n",
    "df['a'].argsort()[::-1][n]\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "ser = pd.Series(np.random.randint(1, 100, 15))\n",
    "\n",
    "# Solution\n",
    "print('ser: ', ser.tolist(), 'mean: ', round(ser.mean()))\n",
    "np.argwhere(ser > ser.mean())[1]"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))\n",
    "\n",
    "# Solution\n",
    "# print row sums\n",
    "rowsums = df.apply(np.sum, axis=1)\n",
    "\n",
    "# last two rows with row sum greater than 100\n",
    "last_two_rows = df.iloc[np.where(rowsums > 100)[0][-2:], :]"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "ser = pd.Series(np.logspace(-2, 2, 30))\n",
    "\n",
    "# Solution\n",
    "def cap_outliers(ser, low_perc, high_perc):\n",
    "    low, high = ser.quantile([low_perc, high_perc])\n",
    "    print(low_perc, '%ile: ', low, '|', high_perc, '%ile: ', high)\n",
    "    ser[ser < low] = low\n",
    "    ser[ser > high] = high\n",
    "    return(ser)\n",
    "\n",
    "capped_ser = cap_outliers(ser, .05, .95)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.random.randint(-20, 50, 100).reshape(10,-1))\n",
    "print(df)\n",
    "\n",
    "# Solution\n",
    "# Step 1: remove negative values from arr\n",
    "arr = df[df > 0].values.flatten()\n",
    "arr_qualified = arr[~np.isnan(arr)]\n",
    "\n",
    "# Step 2: find side-length of largest possible square\n",
    "n = int(np.floor(arr_qualified.shape[0]**.5))\n",
    "\n",
    "# Step 3: Take top n^2 items without changing positions\n",
    "top_indexes = np.argsort(arr_qualified)[::-1]\n",
    "output = np.take(arr_qualified, sorted(top_indexes[:n**2])).reshape(n, -1)\n",
    "print(output)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.arange(25).reshape(5, -1))\n",
    "\n",
    "# Solution\n",
    "def swap_rows(df, i1, i2):\n",
    "    a, b = df.iloc[i1, :].copy(), df.iloc[i2, :].copy()\n",
    "    df.iloc[i1, :], df.iloc[i2, :] = b, a\n",
    "    return df\n",
    "\n",
    "print(swap_rows(df, 1, 2))\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.arange(25).reshape(5, -1))\n",
    "\n",
    "# Solution 1\n",
    "df.iloc[::-1, :]\n",
    "\n",
    "# Solution 2\n",
    "print(df.loc[df.index[::-1], :])\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.arange(25).reshape(5,-1), columns=list('abcde'))\n",
    "\n",
    "# Solution\n",
    "df_onehot = pd.concat([pd.get_dummies(df['a']), df[list('bcde')]], axis=1)\n",
    "print(df_onehot)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\r\n",
    "df = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1))\r\n",
    "\r\n",
    "# Solution\r\n",
    "print('Column with highest row maxes: ', df.apply(np.argmax, axis=1).value_counts().index[0])\r\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "df = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1), columns=list('pqrs'), index=list('abcdefghij'))\r\n",
    "\r\n",
    "# Solution\r\n",
    "import numpy as np\r\n",
    "\r\n",
    "# init outputs\r\n",
    "nearest_rows = []\r\n",
    "nearest_distance = []\r\n",
    "\r\n",
    "# iterate rows.\r\n",
    "for i, row in df.iterrows():\r\n",
    "    curr = row\r\n",
    "    rest = df.drop(i)\r\n",
    "    e_dists = {}  # init dict to store euclidean dists for current row.\r\n",
    "    # iterate rest of rows for current row\r\n",
    "    for j, contestant in rest.iterrows():\r\n",
    "        # compute euclidean dist and update e_dists\r\n",
    "        e_dists.update({j: round(np.linalg.norm(curr.values - contestant.values))})\r\n",
    "    # update nearest row to current row and the distance value\r\n",
    "    nearest_rows.append(max(e_dists, key=e_dists.get))\r\n",
    "    nearest_distance.append(max(e_dists.values()))\r\n",
    "\r\n",
    "df['nearest_row'] = nearest_rows\r\n",
    "df['dist'] = nearest_distance"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1), columns=list('pqrstuvwxy'), index=list('abcdefgh'))\n",
    "df\n",
    "\n",
    "# Solution\n",
    "abs_corrmat = np.abs(df.corr())\n",
    "max_corr = abs_corrmat.apply(lambda x: sorted(x)[-2])\n",
    "print('Maximum Correlation possible for each column: ', np.round(max_corr.tolist(), 2))\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))\n",
    "\n",
    "# Solution 1\n",
    "min_by_max = df.apply(lambda x: np.min(x)/np.max(x), axis=1)\n",
    "\n",
    "# Solution 2\n",
    "min_by_max = np.min(df, axis=1)/np.max(df, axis=1)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))\n",
    "\n",
    "# Solution\n",
    "out = df.apply(lambda x: x.sort_values().unique()[-2], axis=1)\n",
    "df['penultimate'] = out\n",
    "print(df)"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))\n",
    "\n",
    "# Solution Q1\n",
    "out1 = df.apply(lambda x: ((x - x.mean())/x.std()).round(2))\n",
    "print('Solution Q1\\n',out1)\n",
    "\n",
    "# Solution Q2\n",
    "out2 = df.apply(lambda x: ((x.max() - x)/(x.max() - x.min())).round(2))\n",
    "print('Solution Q2\\n', out2)  \n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))\n",
    "\n",
    "# Solution\n",
    "[df.iloc[i].corr(df.iloc[i+1]).round(2) for i in range(df.shape[0])[:-1]]\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.random.randint(1,100, 100).reshape(10, -1))\n",
    "\n",
    "# Solution\n",
    "for i in range(df.shape[0]):\n",
    "    df.iat[i, i] = 0\n",
    "    df.iat[df.shape[0]-i-1, i] = 0"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame({'col1': ['apple', 'banana', 'orange'] * 3,\n",
    "                   'col2': np.random.rand(9),\n",
    "                   'col3': np.random.randint(0, 15, 9)})\n",
    "\n",
    "df_grouped = df.groupby(['col1'])\n",
    "\n",
    "# Solution 1\n",
    "df_grouped.get_group('apple')\n",
    "\n",
    "# Solution 2\n",
    "for i, dff in df_grouped:\n",
    "    if i == 'apple':\n",
    "        print(dff)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,\n",
    "                   'taste': np.random.rand(9),\n",
    "                   'price': np.random.randint(0, 15, 9)})\n",
    "\n",
    "print(df)\n",
    "\n",
    "# Solution\n",
    "df_grpd = df['taste'].groupby(df.fruit)\n",
    "df_grpd.get_group('banana').sort_values().iloc[-2]\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,\n",
    "                   'rating': np.random.rand(9),\n",
    "                   'price': np.random.randint(0, 15, 9)})\n",
    "\n",
    "# Solution\n",
    "out = df.groupby('fruit', as_index=False)['price'].mean()\n",
    "print(out)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df1 = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,\n",
    "                    'weight': ['high', 'medium', 'low'] * 3,\n",
    "                    'price': np.random.randint(0, 15, 9)})\n",
    "\n",
    "df2 = pd.DataFrame({'pazham': ['apple', 'orange', 'pine'] * 2,\n",
    "                    'kilo': ['high', 'low'] * 3,\n",
    "                    'price': np.random.randint(0, 15, 6)})\n",
    "\n",
    "# Solution\n",
    "pd.merge(df1, df2, how='inner', left_on=['fruit', 'weight'], right_on=['pazham', 'pounds'], suffixes=['_left', '_right'])"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df1 = pd.DataFrame({'fruit': ['apple', 'orange', 'banana'] * 3,\n",
    "                    'weight': ['high', 'medium', 'low'] * 3,\n",
    "                    'price': np.arange(9)})\n",
    "\n",
    "df2 = pd.DataFrame({'fruit': ['apple', 'orange', 'pine'] * 2,\n",
    "                    'weight': ['high', 'medium'] * 3,\n",
    "                    'price': np.arange(6)})\n",
    "\n",
    "\n",
    "# Solution\n",
    "print(df1[~df1.isin(df2).all(1)])\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame({'fruit1': np.random.choice(['apple', 'orange', 'banana'], 10),\n",
    "                    'fruit2': np.random.choice(['apple', 'orange', 'banana'], 10)})\n",
    "\n",
    "# Solution\n",
    "np.where(df.fruit1 == df.fruit2)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.random.randint(1, 100, 20).reshape(-1, 4), columns = list('abcd'))\n",
    "\n",
    "# Solution\n",
    "df['a_lag1'] = df['a'].shift(1)\n",
    "df['b_lead1'] = df['b'].shift(-1)\n",
    "print(df)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame(np.random.randint(1, 10, 20).reshape(-1, 4), columns = list('abcd'))\n",
    "\n",
    "# Solution\n",
    "pd.value_counts(df.values.ravel())\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# Input\n",
    "df = pd.DataFrame([\"STD, City    State\",\n",
    "\"33, Kolkata    West Bengal\",\n",
    "\"44, Chennai    Tamil Nadu\",\n",
    "\"40, Hyderabad    Telengana\",\n",
    "\"80, Bangalore    Karnataka\"], columns=['row'])\n",
    "\n",
    "# Solution\n",
    "df_out = df.row.str.split(',|\\t', expand=True)\n",
    "\n",
    "# Make first row as header\n",
    "new_header = df_out.iloc[0]\n",
    "df_out = df_out[1:]\n",
    "df_out.columns = new_header\n",
    "print(df_out)\n"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [],
   "outputs": [],
   "metadata": {}
  }
 ],
 "metadata": {
  "kernelspec": {
   "name": "python3",
   "display_name": "Python 3.9.6 64-bit"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  },
  "interpreter": {
   "hash": "7e9c1221d0e4fe6466a50b4e78c622ed9b09dad7a0a444917045320970a233f5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}