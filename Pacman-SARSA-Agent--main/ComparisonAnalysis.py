import matplotlib.pylab as plt
import pickle, glob, sys, os
import statistics as stat
import numpy as np

class ComparisonData:
    def __init__(self, d1=None, d2=None, x1=None, x2=None, y1=None, y2=None):
        if None not in (d1, d2):
            self.x1, self.y1 = zip(*d1.items())
            self.x2, self.y2 = zip(*d2.items())
        elif None not in (x1, x2, y1, y2):
            self.x1 = x1
            self.x2 = x2
            self.y1 = y1
            self.y2 = y2
        else:
            raise Exception("Invalid comparison data initialization parameters")
    
    def copy(self):
        return ComparisonData(x1=self.x1, x2=self.x2, y1=self.y1, y2=self.y2)

    def cumAvg(self):
        y1 = np.array(self.y1)
        y1 = np.cumsum(y1)/range(1,len(self.y1)+1)
        y2 = np.array(self.y2)
        y2 = np.cumsum(y2)/range(1,len(self.y2)+1)
        newData = self.copy() 
        newData.y1 = y1.tolist()
        newData.y2 = y2.tolist()
        return newData

    def frameAvg(self, framesize):
        accumulated_y1 = []
        for idx in range(len(self.y1)):
            s = max(0, idx-framesize)
            accumulated_y1.append(stat.mean(self.y1[s:idx+1]))
        accumulated_y2 = []
        for idx in range(len(self.y1)):
            s = max(0, idx-framesize)
            accumulated_y2.append(stat.mean(self.y2[s:idx+1]))
        newData = self.copy()
        newData.y1 = accumulated_y1
        newData.y2 = accumulated_y2
        return newData
    
    def convCheck(self, threshold=10**-6):
        """
        Does not work very well, feel free to replace
        """
        conv_points = [-1, -1]
        for idx in range(1, len(self.y1)):
            if abs(self.y1[idx] - self.y1[idx-1]) < threshold:
                conv_points[0] = idx
        for idx in range(1, len(self.y2)):
            if abs(self.y2[idx] - self.y2[idx-1]) < threshold:
                conv_points[1] = idx
        return conv_points

class ComparisonPlotter:
    def __init__(self, xl, yl, l1, l2, t):
        self.t = t
        self.l1 = l1
        self.l2 = l2
        self.xl = xl
        self.yl = yl

    def plot(self, data: ComparisonData):
        plt.plot(data.x1, data.y1, color='r', label=self.l1)
        plt.plot(data.x2, data.y2, color='g', label=self.l2)
        plt.xlabel(self.xl)
        plt.ylabel(self.yl)
        plt.title(self.t)
        plt.legend()
    
    def show(self):
        plt.show()

def analyze(comparisonOuput):
    #output = {"SARSA": (SARSAWeights, SARSAgames), "Q-Agent": (QAgentWeights, QAgentgames)}
    SARSAWeights, SARSAgames = comparisonOuput["SARSA"]
    QAgentWeights, QAgentgames = comparisonOuput["Q-Agent"]
    if "layout" in comparisonOuput.keys():
        layout = comparisonOuput["layout"]
        print("data layout: %s" % layout)
    if "seed" in comparisonOuput.keys():
        seed = comparisonOuput["seed"]
        print("data seed: %d" % seed)

    compData = ComparisonData(d1=SARSAgames, d2=QAgentgames)
    plot = ComparisonPlotter('Episodes', "Average Score", "SARSA", "QAgent", "Comparison of Learning Agents")

    plot.plot(compData)
    plot.show()

    plot.plot(compData.cumAvg())
    plot.show()

    plot.plot(compData.frameAvg(30))
    plot.show()

    cpts = compData.cumAvg().convCheck(threshold=1)
    if -1 in cpts:
        print("One or more algorithms did not reach the convergence threshold")
    else:
        print("SARSA converged at episode %d\nQ-Agent Converged at episode %d" % tuple(cpts))

if __name__ == "__main__":
    if len(sys.argv) == 2:
        ouput_file = sys.argv[1]
        if not os.path.exists(ouput_file):
            raise Exception("File %s does not exist" % sys.argv[1])
    else:
        ouput_files = glob.glob('./outputs/comp*')
        ouput_file = max(ouput_files, key=os.path.getctime)

    print("Loading output file %s" % ouput_file) 

    with open(ouput_file, 'rb') as f:
        ouput = pickle.load(f)
        analyze(ouput)
