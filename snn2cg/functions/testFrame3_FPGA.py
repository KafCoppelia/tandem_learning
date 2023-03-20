COREX = 16
COREY = 16
COREYBIT = 5

CHIPID = 0 
FRAMEHEAD = (1 << 2) + 2

SRAMBEG = 0
FRAMENUM = 4096 * 4

FRAMEFORMAT = (FRAMEHEAD << 60) + (CHIPID << 50) + (SRAMBEG << 20) + (1 << 19) + FRAMENUM

def genTestFrame4():
    frames = list()
    for i in range(COREX):
        for j in range(COREY):
            coreId = (i << COREYBIT) + j
            frame = "{:064b}".format(FRAMEFORMAT + (coreId << 40))
            frames.append(frame)
    with open("testFrame3.txt",'w') as f:
        f.write("\n".join(frames))
    return 

if __name__ == "__main__":
    genTestFrame4()