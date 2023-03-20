
import sys
sys.path.append("..")
sys.path.append(".")
import os
from runtime import setOnChipNetwork, runOnChipNetwork
def runSimulator(frameDir, debugTime):
    configPath = os.path.join(frameDir, "config.txt")
    inputPath = os.path.join(frameDir, "input_0.txt")
    outputPath = os.path.join(frameDir, "output_0.txt")
    debugPath = os.path.join(frameDir, "out.txt")
    simulator = setOnChipNetwork(configPath)
    if os.path.exists(debugPath) and debugTime > 0:
        simulator.debugNeuronParas(debugPath, debugTime)
    runOnChipNetwork(simulator, inputPath, outputPath, None)

if __name__ == "__main__":
    # frameDir = "output/debug/test/example_net0withbias1"
    # frameDir = "/home/xiuping/projects/SNN/PAIFlow/output/test/frames"
    # frameDir = "/home/xiuping/projects/SNN/PAIFlow/debug_net0bias/frames"
    frameDir = "/home/xiuping/projects/SNN/PAIFlow/SNN2CG/output/ASIC_LongRun_255_25"
    # frameDir = "output/debug/test/example0withoutBias"
    # frameDir = "/home/xiuping/projects/SNN/PAIFlow/SNN2CG/output/files"
    runSimulator(frameDir, 54)