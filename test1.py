# coding=utf-8

import sys
import sqlanydb


# conn = sqlanydb.connect(uid='sa', pwd='', eng='iso_1', dbn='demo' )

class Test1:
    def __init__(self):
        self.y = []
        self.x = 100


t = Test1()
t.x = 123
t.y = [1, 23, 232]
# 反射：
print(hasattr(t, "x"))
print(getattr(t, "x"))
print(dir(t))
tatts = dir(t)
for at in tatts:
    if (at.find("__") < 0) and (hasattr(t,at)):
        setattr(t,"x",1000)
        print(getattr(t,at))
#判断变量类型的函数
def typeof(variate):
    type=None
    if isinstance(variate,int):
        type = "int"
    elif isinstance(variate,str):
        type = "str"
    elif isinstance(variate,float):
        type = "float"
    elif isinstance(variate,list):
        type = "list"
    elif isinstance(variate,tuple):
        type = "tuple"
    elif isinstance(variate,dict):
        type = "dict"
    elif isinstance(variate,set):
        type = "set"
    return type
# 返回变量类型
def getType(variate):
    arr = {"int":"整数","float":"浮点","str":"字符串","list":"列表","tuple":"元组","dict":"字典","set":"集合"}
    vartype = typeof(variate)
    if not (vartype in arr):
        return "未知类型"
    return arr[vartype]

#判断变量是否为整数
money=120
print("{0}是{1}".format(money,getType(money)))
#判断变量是否为字符串
money="120"
print("{0}是{1}".format(money,getType(money)))
money=12.3
print("{0}是{1}".format(money,getType(money)))
#判断变量是否为列表
students=['studentA']
print("{0}是{1}".format(students,getType(students)))
#判断变量是否为元组
students=('studentA','studentB')
print("{0}是{1}".format(students,getType(students)))
#判断变量是否为字典
dictory={"key1":"value1","key2":"value2"}
print("{0}是{1}".format(dictory,getType(dictory)))
#判断变量是否为集合
apple={"apple1","apple2"}
print("{0}是{1}".format(apple,getType(apple)))
sys.exit()
for i in range(10):
    t.y.append(i)
    print(t.y)
print(sys.platform)
print("dfdsff")
print(eval("10**100"))
sys.exit()
if __debug__:
    print('debug')
for i in range(10):
    print(i)

a = np.array(([3, 2, 1], [2, 5, 7], [4, 7, 8]))

itemindex = np.argwhere(a == 7)

print(itemindex)

print('dfsdfsdfdsfsdfsfsdfdsf', a)
b = np.array([10.2, 20, 34, 20, 50, 60, 449])
c = np.argwhere(b == 11120)
print('aaaaaa', c.flatten())

print('this is a test:\n\r', True, "hahah")
x = Decimal('2.22')
getcontext().prec
# sys.exit()
net = buildNetwork(2, 4, 1)
print(net)

NetworkWriter.writeToFile(net, 'testNet.xml')
net = NetworkReader.readFrom('testNet.xml')
print(net)

z = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])
print(str(z.shape))
print(z.reshape(-1, 1))
z1 = z.reshape(-1, 1)
print(z1.shape)
print(z.reshape(2, -1))
print(z.reshape(-1))
