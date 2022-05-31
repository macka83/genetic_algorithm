#!/usr/bin/env python
# coding: utf-8

# In[16]:


import copy


# In[17]:


edges = [('in', 'out0',1),
         ('mid0', 'out0',2),
         ('in','mid0',3),
         ('mid0','mid1',4), 
         ('mid1','mid0',5),
         ('in2','mid1',6),
         ('mid1','out1',7),
         ('mid0','mid0',4)]


# In[25]:


edges = [('in0', 'out0', 1),
        ('mid1', 'out1',2),
        ('mid1', 'out2', 3),
        ('in1','out1', 4),
        ('mid1', 'mid0',5),
        ('mid0', 'mid1',6),
        ('in2', 'mid0',8)]


# https://github.com/davidrmiller/biosim4/issues/56

# ![148993699-99708c7f-81b2-46bb-a425-921f42bbedab.png](attachment:148993699-99708c7f-81b2-46bb-a425-921f42bbedab.png)

# In[26]:


dic = {}
for el in edges:
    if el[1] in dic:
        dic[el[1]].update({el[0]:{el[2]}})
    else:
        dic[el[1]] = {el[0]:{el[2]}}


# In[27]:


for key in dic:
    if 'mid' in key:
        for key_0 in a[key]:
            if 'mid' in key_0 and key_0 in dic:
                b = copy.copy(dic[key_0])
                del b[key]
                dic[key][key_0] = b


# In[28]:


dic


# In[ ]:




