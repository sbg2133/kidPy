import numpy as np

filename = "trans_func_1535061465-Aug-23-2018-14-57-45.dir.npy"

# load transfunc
transfunc = np.load(filename)
#make backup
np.save(filename +".backup_1",transfunc)

print(transfunc.shape)

editing = True

while editing:
    index = np.int(raw_input("What index resonator to edit? "))
    power_factor = np.float(raw_input("What power factor to adjust by? "))
    print(index,power_factor)
    print("old trans factor",transfunc[index])
    transfunc[index] = transfunc[index]*np.sqrt(power_factor)
    print("new trans factor",transfunc[index])
    done = raw_input("Are you finished y/n ? ")
    if done == "y":
        editing = False


np.save(filename,transfunc)
