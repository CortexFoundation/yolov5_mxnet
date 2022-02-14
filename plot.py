import numpy as np
from matplotlib import pyplot as plt

file_name = "./yolov5s_log.log"
f =open(file_name, "r")

lines = f.readlines()
ITER = []
LOSS = []
LBOX = []
LOBJ = []
LCLS = []
for i, line in enumerate(lines):
    #print(line)
    if len(line.split("iter ")) < 2:
        continue
    if i % 4 < 3:
        continue
    numb = line.split("iter ")[1].split(":")[0]
    #print(numb)
    ITER.append(eval(numb))
    try:
        loss = line.split("loss = ")[1].split(",")[0][0:-1]
    except:
        break
    #print(loss)
    LOSS.append(eval(loss))
    
    lbox = line.split("lbox = ")[1].split(",")[0][0:-1]
    #print(lbox)
    LBOX.append(eval(lbox))
    
    lobj = line.split("lobj = ")[1].split(",")[0][0:-1]
    #print(lobj)
    LOBJ.append(eval(lobj)) 
    
    lcls = line.split("lcls = ")[1].split(". bs=")[0][0:-1]
    #print(lcls)
    LCLS.append(eval(lcls))
    
    #if eval(numb) == 100:
    #    break


plt.close()
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title(file_name)
plt.plot(np.array(LBOX))
plt.plot(np.array(LOBJ))
plt.plot(np.array(LCLS))
plt.plot(np.array(LOSS))
plt.legend(["box","obj","cls","loss"])
plt.show()
  
