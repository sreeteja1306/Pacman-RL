import matplotlib.pylab as plt
import pickle, glob, sys, os
import statistics as stat
import numpy as np
import pandas as pd
from scipy import stats
from IPython import embed


class TrainingAnalysys:

    def __init__(self, data,frame = 10, Agent = None):
        if Agent is not None:
            self.Agent = Agent
        self.weights = data[0]
        self.Rewards = list(data[1].values())
        self.y = self.Rewards
        self.frameSize = frame
        self.x = data[1].keys()
        self.cumulativeAverage = self.cumAvg()
        self.accumulatedAverage = self.frameAvg(frame)

        self.CP , self.CR = self.convergencePoint(self.cumulativeAverage,20,5)


    def cumAvg(self):
        cumulativeAverage = np.array(self.y)
        cumulativeAverage = np.cumsum(cumulativeAverage) / range(1, len(self.y) + 1)
        return cumulativeAverage.tolist()

    def getCumAvg(self):
        return self.cumulativeAverage

    def frameAvg(self, framesize):
        accumulatedAverage = []
        for idx in range(len(self.y)):
            s = max(0, idx - framesize)
            accumulatedAverage.append(stat.mean(self.y[s:idx + 1]))
        return accumulatedAverage

    def getAccumulatedAverage(self,frame = None):
        if frame is not None:
            if frame == self.frameSize:
                return self.accumulatedAverage
            else:
                self.frameSize = frame
                self.accumulatedAverage = self.frameAvg(frame)
                return self.accumulatedAverage
        else:
            return self.accumulatedAverage

    def convergencePoint(self,data,WindowSize = 5,threshold = 5):
        for convergenceIteration in range(WindowSize//2,len(data)-(WindowSize//2)):
            s = stat.stdev(data[convergenceIteration-(WindowSize//2):convergenceIteration+WindowSize])
            if s < threshold:
                return convergenceIteration , data[convergenceIteration]
        return 0,0
    
    def plotGraph(self,axes):
        axes[0].plot(self.x, self.cumulativeAverage, label=self.Agent)
        #axes[1].plot(self.x, self.accumulatedAverage, label='accumulatedAverage')
        #axes.plot(self.x, self.Rewards, color='k', label='reward')
        axes[0].plot(self.CP, self.CR, 'b*')

def conv_analyze(RunResults, what: str):
    if "layout" in RunResults.keys():
        layout = RunResults["layout"]
        print("data layout: %s" % layout)
    if "seed" in RunResults.keys():
        seed = RunResults["seed"]
        print("data seed: %d" % seed)
    out = {'PacmanSARSAAgent': { 'SpookedExtractor' : [],
                                 'SimpleExtractor' : []
                                },
            'ApproximateQAgent':{'SpookedExtractor': [],
                                 'SimpleExtractor': []
                             }
        }
    if what == "ALL":
        for run in RunResults['runs']:
            sasp = TrainingAnalysys(RunResults['runs'][run]['PacmanSARSAAgent']['SpookedExtractor'],Agent='SARSA-Spooked')
            out['PacmanSARSAAgent']['SpookedExtractor'].append((sasp.CP, sasp.CR))
            aqsp = TrainingAnalysys(RunResults['runs'][run]['ApproximateQAgent']['SpookedExtractor'],Agent='QAgent-Spooked')
            out['ApproximateQAgent']['SpookedExtractor'].append((aqsp.CP, aqsp.CR))
            sase = TrainingAnalysys(RunResults['runs'][run]['PacmanSARSAAgent']['SimpleExtractor'], Agent='SARSA-Simple')
            out['PacmanSARSAAgent']['SimpleExtractor'].append((sase.CP, sase.CR))
            aqse = TrainingAnalysys(RunResults['runs'][run]['ApproximateQAgent']['SimpleExtractor'], Agent='QAgent-Simple')
            out['ApproximateQAgent']['SimpleExtractor'].append((aqse.CP, aqse.CR))
    elif what == "CONVP":
        for run in RunResults['runs']:
            sasp = TrainingAnalysys(RunResults['runs'][run]['PacmanSARSAAgent']['SpookedExtractor'],Agent='SARSA-Spooked')
            out['PacmanSARSAAgent']['SpookedExtractor'].append((sasp.CP))
            aqsp = TrainingAnalysys(RunResults['runs'][run]['ApproximateQAgent']['SpookedExtractor'],Agent='QAgent-Spooked')
            out['ApproximateQAgent']['SpookedExtractor'].append((aqsp.CP))
            sase = TrainingAnalysys(RunResults['runs'][run]['PacmanSARSAAgent']['SimpleExtractor'], Agent='SARSA-Simple')
            out['PacmanSARSAAgent']['SimpleExtractor'].append((sase.CP))
            aqse = TrainingAnalysys(RunResults['runs'][run]['ApproximateQAgent']['SimpleExtractor'], Agent='QAgent-Simple')
            out['ApproximateQAgent']['SimpleExtractor'].append((aqse.CP))
    elif what == "CONVR":
        for run in RunResults['runs']:
            sasp = TrainingAnalysys(RunResults['runs'][run]['PacmanSARSAAgent']['SpookedExtractor'],Agent='SARSA-Spooked')
            out['PacmanSARSAAgent']['SpookedExtractor'].append((sasp.CR))
            aqsp = TrainingAnalysys(RunResults['runs'][run]['ApproximateQAgent']['SpookedExtractor'],Agent='QAgent-Spooked')
            out['ApproximateQAgent']['SpookedExtractor'].append((aqsp.CR))
            sase = TrainingAnalysys(RunResults['runs'][run]['PacmanSARSAAgent']['SimpleExtractor'], Agent='SARSA-Simple')
            out['PacmanSARSAAgent']['SimpleExtractor'].append((sase.CR))
            aqse = TrainingAnalysys(RunResults['runs'][run]['ApproximateQAgent']['SimpleExtractor'], Agent='QAgent-Simple')
            out['ApproximateQAgent']['SimpleExtractor'].append((aqse.CR))
    return out

def t_test():
    pass


def calc_tval(sample1, sample2, ttype: str = 'two_tailed'):
    pass
    #v = stats.ttest_ind(a=sample1, b=sample2, alternative=ttype)
    return 

def do_algorithm_ttest(data, extractor):
    # compare convergence of agents on both extractors
    convp_s = []
    convr_s = []

    convp_q = []
    convr_q = []

    for cp, cr in data['PacmanSARSAAgent'][extractor]:
        convp_s.append(cp)
        convr_s.append(cr)

    for cp, cr in data['ApproximateQAgent'][extractor]:
        convp_q.append(cp)
        convr_q.append(cr)
    
    convp_tval = stats.ttest_ind(a=np.array(convp_s), b=np.array(convp_q), alternative='two-sided')
    convr_tval = stats.ttest_ind(a=np.array(convr_s), b=np.array(convr_q), alternative='two-sided')
    return (convp_tval, convr_tval)

def do_extractor_ttest(data, agent):
    # compare convergence of extractors
    convp_sp = []
    convr_sp = []

    convp_si = []
    convr_si = []

    for cp, cr in data[agent]['SpookedExtractor']:
        convp_sp.append(cp)
        convr_sp.append(cr)
    
    for cp, cr in data[agent]['SimpleExtractor']:
        convp_si.append(cp)
        convr_si.append(cr)

    convp_tval = stats.ttest_ind(a=np.array(convp_si), b=np.array(convp_sp), alternative='less')
    convr_tval = stats.ttest_ind(a=np.array(convr_sp), b=np.array(convr_si), alternative='greater')
    return (convp_tval, convr_tval)

def t_test_convergence_analysis():
    spooked_conv_agent_out = dict()
    test = pd.DataFrame()
    simple_conv_agent_out = dict()
    sarsa_x_conv_agent_out = dict()
    qagent_x_conv_agent_out = dict()
    layout_names = []
    for ouput_file in ouput_files:
        with open(ouput_file, 'rb') as f:
            output = pickle.load(f)
            layout_name = output['layout_name']
            print('layout', layout_name)
            layout_names.append(layout_name)

            conv_data = conv_analyze(output, "ALL")

            def check_sig(pvalue): return pvalue < 0.05

            convp_tval, convr_tval = do_algorithm_ttest(conv_data, 'SpookedExtractor')
            spooked_conv_agent_out[layout_name] =  {
                                                    'convp-tval': convp_tval.statistic, 'convp-pval': convp_tval.pvalue, 
                                                    'convr-tval': convr_tval.statistic, 'convr-pval': convr_tval.pvalue
                                                    }

            convp_tval, convr_tval = do_algorithm_ttest(conv_data, 'SimpleExtractor')
            simple_conv_agent_out[layout_name] =  {
                                                    'convp-tval': convp_tval.statistic, 'convp-pval': convp_tval.pvalue, 
                                                    'convr-tval': convr_tval.statistic, 'convr-pval': convr_tval.pvalue
                                                    }

            qagent_x_conv_agent_out[layout_name] = {'convp-tval': [], 'convp-pval': [], 'convr-tval': [], 'convr-pval': []}

            convp_tval, convr_tval = do_extractor_ttest(conv_data, 'PacmanSARSAAgent')
            sarsa_x_conv_agent_out[layout_name] =  {
                                                    'convp-tval': convp_tval.statistic, 'convp-pval': convp_tval.pvalue, 
                                                    'convr-tval': convr_tval.statistic, 'convr-pval': convr_tval.pvalue
                                                    }

            convp_tval, convr_tval = do_extractor_ttest(conv_data, 'ApproximateQAgent')
            qagent_x_conv_agent_out[layout_name] =  {
                                                    'convp-tval': convp_tval.statistic, 'convp-pval': convp_tval.pvalue, 
                                                    'convr-tval': convr_tval.statistic, 'convr-pval': convr_tval.pvalue
                                                    }
    
    def output_as_csv(data: dict, output_file):
        df = pd.DataFrame(data).transpose()
        df.to_csv(output_file)
    
    output_as_csv(spooked_conv_agent_out, "./stat-results/spooked_agent_convergence.csv")
    output_as_csv(simple_conv_agent_out, "./stat-results/simple_agent_convergence.csv")
    output_as_csv(sarsa_x_conv_agent_out, "./stat-results/sarsa_extractor_convergence.csv")
    output_as_csv(qagent_x_conv_agent_out, "./stat-results/qagent_extractor_convergence.csv")




if __name__ == "__main__":
    if len(sys.argv) == 2:
        ouput_file = sys.argv[1]
        if not os.path.exists(ouput_file):
            raise Exception("File %s does not exist" % sys.argv[1])
    else:
        ouput_files = glob.glob('./results/*')
        ouput_file = max(ouput_files, key=os.path.getctime)

    print("Loading output file %s" % ouput_file)

    for ouput_file in ouput_files:

        with open(ouput_file, 'rb') as f:
            output = pickle.load(f)
            for runs in output['runs']:
                fig,axes = plt.subplots(1,1)
                TrainingAnalysys(output['runs'][runs]['PacmanSARSAAgent']['SpookedExtractor'],Agent='SARSA-Spooked').plotGraph([axes])
                TrainingAnalysys(output['runs'][runs]['PacmanSARSAAgent']['SimpleExtractor'],Agent='SARSA-Simple').plotGraph([axes])
                TrainingAnalysys(output['runs'][runs]['ApproximateQAgent']['SpookedExtractor'],Agent='QAgent-Spooked').plotGraph([axes])
                TrainingAnalysys(output['runs'][runs]['ApproximateQAgent']['SimpleExtractor'],Agent='QAgent-Simple').plotGraph([axes])
                plt.legend()
                axes.set_title(output['layout_name'].split('.')[0].split('/')[1])
                plt.show()
    
    layout_sp_agent_data = dict()
    for ouput_file in ouput_files:
        with open(ouput_file, 'rb') as f:
            output = pickle.load(f)
            conv_data = conv_analyze(output, 'CONVP')
            layout_path = output['layout_name']
            layout_name = os.path.basename(layout_path).replace('.lay', '')
            layout_sp_agent_data[f'{layout_name} (SARSA)'] = conv_data['PacmanSARSAAgent']['SpookedExtractor']
            layout_sp_agent_data[f'{layout_name} (QAgent)'] = conv_data['ApproximateQAgent']['SpookedExtractor']
    pd.DataFrame(layout_sp_agent_data).to_csv("stat-results/conv_point_data.csv")
    # analyze and generate t_test csv data
    t_test_convergence_analysis()