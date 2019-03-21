# -*- coding: utf-8 -*-
old = []
new = []

with open('labels_v2.txt', 'rb') as f:
    for i in f.readlines():
        old.append(i.strip().split(':'))
with open('labels_337_4_10.txt', 'rb') as f:
    for i in f.readlines():
        new.append(i.strip().split(':'))
for index, i in enumerate(new):
    for j in old:
        if i[1] == j[1]:
            new[index].extend(j[2:])
item = ['其他', '其他', '-']
for index,i in enumerate(new):
    print(len(i))
    if len(i) < 3:
        new[index].extend(item)
       

with open('labels_v2_temp.txt', 'w') as f:
    for i in new:
        if len(i) == 4:
            f.write('{}:{}:{}:{}\n'.format(i[0],i[1],i[2],i[3]))
        else:
            f.write('{}:{}:{}:{}:{}\n'.format(i[0],i[1],i[2],i[3],i[4]))