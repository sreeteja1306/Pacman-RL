import random, pickle, tqdm, util, time, sys, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from ComparisonAnalysis import analyze
from collections import OrderedDict
from statistics import mean
import pacman as pc

def runTraining(layout, pacman, ghosts, display, numGames, record, numTraining=0, catchExceptions=False, timeout=30):
    import __main__
    __main__.__dict__['_display'] = display

    rules = pc.ClassicGameRules(timeout)
    weights_list = []
    def callback(game_state):
        weights_list.append(game_state.agents[0].getWeights().copy())

    for i in tqdm.tqdm(range(numGames)):
        beQuiet = True
        if beQuiet:
                # Suppress output and graphics
            import textDisplay
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = True
        else:
            gameDisplay = display
            rules.quiet = False
        game = rules.newGame(layout, pacman, ghosts,
                             gameDisplay, beQuiet, catchExceptions)
        game.run(callback=callback)
    return weights_list[:-1]

def runTrials(layout, pacman, ghosts, display, numGames, record, numTraining=0, catchExceptions=False, timeout=10, weight={}):
    import __main__
    __main__.__dict__['_display'] = display

    rules = pc.ClassicGameRules(timeout)
    games = []
    score = []

    for _ in range(numGames):
        beQuiet = True
        if beQuiet:
                # Suppress output and graphics
            import textDisplay
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = True
        else:
            gameDisplay = display
            rules.quiet = False
        game = rules.newGame(layout, pacman, ghosts,
                             gameDisplay, beQuiet, catchExceptions)
        game.agents[0].weights = weight
        game.run()
        games.append(game)
        score.append(game.state.getScore())
    return mean(score)

def train(episode, agent):
    print("Training", agent)
    temp = ['-x',str(episode),'-n',str(episode+1),'-p', agent]
    temp.extend(sys.argv[1:])
    args = pc.readCommand(temp)  # Get game components based on input
    return runTraining(**args)

def trial(trials, agent, weight):
    temp = ['-x',str(0),'-n',str(trials),'-p', agent]
    temp.extend(sys.argv[1:])
    args = pc.readCommand(temp)  # Get game components based on input
    return runTrials(**args, weight=weight)

def t_task(training_episodes, agent):
    return agent, train(training_episodes, agent)
def s_task(iter, weight, trial_episodes):
    return iter, trial(trial_episodes, 'PacmanSARSAAgent', weight)
def q_task(iter, weight, trial_episodes):
    return iter, trial(trial_episodes, 'ApproximateQAgent', weight)

if __name__ == '__main__':
    training_episodes = 500
    trial_episodes = 5
    seed = random.randint(0, 1<<64)
    random.seed(seed)
    os.environ["beQuiet"] = "1"

    print("Performing comparison test on %d training episodes and %d trials" % (training_episodes, trial_episodes))

    SARSAWeights = []
    QAgentWeights = []
    with ProcessPoolExecutor() as executor:
        futures = []

        futures += [executor.submit(t_task, training_episodes, 'PacmanSARSAAgent')]
        futures += [executor.submit(t_task, training_episodes, 'ApproximateQAgent')]

        for c in as_completed(futures):
            agent, weights = c.result()
            if agent == 'PacmanSARSAAgent':
                SARSAWeights = weights
            elif agent == 'ApproximateQAgent':
                QAgentWeights = weights


    # Prepend empty feature vectors for initial trials
    SARSAWeights = [util.Counter()] + SARSAWeights
    QAgentWeights = [util.Counter()] + QAgentWeights

    SARSAgames = OrderedDict()
    QAgentgames = OrderedDict()

    with tqdm.tqdm(total=len(SARSAWeights)*trial_episodes, desc="SARSA Trials") as pbar:
        with ProcessPoolExecutor() as executor:
            futures = []
            iter: int = 0
            for iter, weight in enumerate(SARSAWeights):
                futures += [executor.submit(s_task, iter, weight, trial_episodes)]
            for c in as_completed(futures):
                iter, res = c.result()
                SARSAgames[iter] = res
                pbar.update(n=trial_episodes)

    with tqdm.tqdm(total=len(QAgentWeights)*trial_episodes, desc="Q-Agent Trials") as pbar:
        with ProcessPoolExecutor() as executor:
            for iter, weight in enumerate(QAgentWeights):
                futures += [executor.submit(q_task, iter, weight, trial_episodes)]
            for c in as_completed(futures):
                iter, res = c.result()
                QAgentgames[iter] = res
                pbar.update(n=trial_episodes)

    # Sort since the parallelized results will be non-sequential
    SARSAgames = OrderedDict(sorted(SARSAgames.items()))
    QAgentgames = OrderedDict(sorted(QAgentgames.items()))

    #layout = pc.readCommand(sys.argv[1:])["layout"]
    layout = pc.readCommand(['-p', 'PacmanSARSAAgent'] + sys.argv[1:])["layout"].name
    output = {"SARSA": (SARSAWeights, SARSAgames), "Q-Agent": (QAgentWeights, QAgentgames), "seed": seed, "layout": layout}
    if not os.path.exists ("./outputs"): os.mkdir("./outputs")
    output_file = f"./outputs/comp-{training_episodes}-{trial_episodes}-{int(time.time())}"
    with open(output_file, "wb+") as f:
        pickle.dump(output, f)
    print("Output dumped to %s" % output_file)
    print("You may view the analysis later by running `python ComparisonAnalysis.py %s`" % output_file)
    analyze(output)
