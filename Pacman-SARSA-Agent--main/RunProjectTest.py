import random, pickle, tqdm, util, time, sys, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from ComparisonAnalysis import analyze
from collections import OrderedDict
from statistics import mean
from IPython import embed
from copy import deepcopy
import pacman as pc

def runTraining(layout, pacman, ghosts, display, numGames, record, numTraining=0, catchExceptions=False, timeout=30):
    import __main__
    __main__.__dict__['_display'] = display

    rules = pc.ClassicGameRules(timeout)
    weights_list = []
    def callback(game_state):
        weights_list.append(game_state.agents[0].getWeights().copy())

    for i in range(numGames):
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

def runTrials(layout, pacman, ghosts, display, numGames, record, numTraining=0, catchExceptions=False, timeout=20, weight={}):
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

def train(episode, agent, extractor):
    #print("Training %s with %s" % (agent, extractor))
    temp = ['-x',str(episode),'-n',str(episode+1),'-p', agent, '-a', f'extractor={extractor}']
    temp.extend(sys.argv[1:])
    args = pc.readCommand(temp)  # Get game components based on input
    return runTraining(**args)

def trial(trials, agent, extractor, weight):
    temp = ['-x',str(0),'-n',str(trials),'-p', agent, '-a', f'extractor={extractor}']
    temp.extend(sys.argv[1:])
    args = pc.readCommand(temp)  # Get game components based on input
    return runTrials(**args, weight=weight)

# task definitions for multiprocessing
def t_task(iter, training_episodes, agent, extractor):
    return iter, agent, extractor, train(training_episodes, agent, extractor)
def r_task(iter, weight, agent, extractor, trial_episodes):
    return iter, agent, extractor, trial(trial_episodes, agent, extractor, weight)

def run_proj_training(exp_runs, training_episodes):
    runs = dict()
    with tqdm.tqdm(total=exp_runs*training_episodes*4, desc="Training") as pbar:
        with ProcessPoolExecutor() as executor:
            futures = []
            for exp in range(exp_runs):
                runs[exp] = {
                    "PacmanSARSAAgent": {
                        "SpookedExtractor": list(),
                        "SimpleExtractor": list()
                    },
                    "ApproximateQAgent": {
                        "SpookedExtractor": list(),
                        "SimpleExtractor": list()
                    }
                }
                futures += [executor.submit(t_task, exp, training_episodes, 'PacmanSARSAAgent', 'SimpleExtractor')]
                futures += [executor.submit(t_task, exp, training_episodes, 'ApproximateQAgent', 'SimpleExtractor')]
                futures += [executor.submit(t_task, exp, training_episodes, 'PacmanSARSAAgent', 'SpookedExtractor')]
                futures += [executor.submit(t_task, exp, training_episodes, 'ApproximateQAgent', 'SpookedExtractor')]

            for c in as_completed(futures):
                exp, agent, extractor, weights = c.result()
                # Prepend empty feature vectors for initial trial
                weights = [util.Counter()] + weights
                runs[exp][agent][extractor] = [weights, OrderedDict()]
                pbar.update(n=training_episodes)
    return runs

def run_proj_trials(trial_episodes: int, exp_runs, training_episodes, training_data: dict):
        trial_data = deepcopy(training_data)
        with tqdm.tqdm(total=((trial_episodes+1)*training_episodes*exp_runs*4), desc="Trialing Weights") as pbar:
            for run_idx, run_data in trial_data.items():
                with ProcessPoolExecutor() as executor:
                    futures = []
                    for agent, agent_data in run_data.items():
                        for extractor, extractor_data in agent_data.items():
                            weights, _ = extractor_data
                            for iter, weight in enumerate(weights):
                                futures += [executor.submit(r_task, iter, weight, agent, extractor, trial_episodes)]
                    for c in as_completed(futures):
                        iter, agent, extractor, data = c.result()
                        _, trial_dict = trial_data[run_idx][agent][extractor]
                        trial_dict[iter] = data
                        pbar.update(trial_episodes)

        # Sort since the parallelized results will be non-sequential
        for run_idx, run_data in trial_data.items():
                for agent, agent_data in run_data.items():
                    for extractor, extractor_data in agent_data.items():
                        weight, trials = extractor_data
                        extractor_data[1] = OrderedDict(sorted(trials.items()))
        return trial_data

def main(exp_runs, training_episodes, trial_episodes, layout):
    # seeding used to work to perfectly reproduce a set of results
    # that was before multiprocesing madness, it may or may not still work
    # but regardless, it is stored in case it is somehow useful
    seed = random.randint(0, 1<<64)
    random.seed(seed)

    print("Performing test %d times on for %d training episodes and %d trials" % (exp_runs, training_episodes, trial_episodes))

    results = {
        'layout_name': layout,
        'seed': seed,
        'Number of runs': exp_runs,
        'runs': None
    }
    training_data: dict = run_proj_training(exp_runs, training_episodes)
    results['runs'] = run_proj_trials(trial_episodes, exp_runs, training_episodes, training_data)

    if not os.path.exists ("./outputs"): os.mkdir("./outputs")
    layout_name = os.path.basename(layout).replace('.lay', '')
    output_file = f"./outputs/{layout_name}_results_{int(time.time())}"
    with open(output_file, "wb+") as f:
        pickle.dump(results, f)

    print("Output dumped to %s" % output_file)

    embed()

if __name__ == '__main__':
    layout = pc.readCommand(['-p', 'PacmanSARSAAgent'] + sys.argv[1:])["layout"].name
    expruns = 25
    training_episodes = 500
    trial_episodes = 5
    os.environ["beQuiet"] = "1"
    main(expruns, training_episodes, trial_episodes, layout)
