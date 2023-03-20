import torch
import os
import argparse

def parseArg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape", default = "3,32,32", type = str,
        help = 'shape of inputs'
    )
    parser.add_argument(
        "--num", default = "10", type = int,
        help = 'number of random data'
    )
    parser.add_argument(
        "--time", default = 100, type = int,
        help = 'time steps'
    )
    parser.add_argument(
        "--dir", default = "./tmp", type = str,
        help = 'Dir to store random data'
    )
    parser.add_argument(
        '--mode', default='snn', type=str,
        help='input mode'
    )
    args = parser.parse_args()
    return args

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def genRandomData(
    dataNum, dataShape, timeStep, inputMode, dir
):
    mkdir(dir)
    for i in range(dataNum):
        data = (torch.rand(timeStep, *dataShape) < 0.1).float()
        if inputMode == 'ann':
            data = data * 64
        filePath = os.path.join(dir, f"data_{i}.pth")
        torch.save(data,filePath)


if __name__ == "__main__":
    args = parseArg()
    dataShape = [int(s) for s in args.shape.split(",")]
    genRandomData(args.num, dataShape, args.time, args.mode, args.dir)