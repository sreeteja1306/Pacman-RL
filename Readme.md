# Reinforcement Learning Project, True Online SARSA

## Pip packages

To run some portions of the project, you may be missing some pip dependencies, to easily install all required pip packages, run the following command

`pip install -r requirements.txt`


## Agent and Extractor
Our implementation of the True online SARSA agent can be found in `SARASAAgents.py`

Our modified feature extractor is located in `featureExtractors.py`

To run a simple test for few games of training using our SARSA agent and extractor, run the follwing command

`python pacman.py -x 5 -n 10 -p PacmanSARSAAgent -l mediumClassic -a extractor=SpookedExtractor`

## Layout Generation
Our random layout generator can be found in `LayoutGenerator.py`

It uses a Growing Tree maze generation algorithm to guarantee the layout is fully reachable

To run the layout generator, run the following command:

`python LayoutGenerator.py`

The parameters of the maze can be tweaked by modifying the variables at the bottom of the file

It is currently in the same configuration as was used to generate the layouts used for our analysis

i.e. it will generate 4 layouts with varying sparseness and complexity, then one medium complex layout

The layout it generates will be displayed for a short time output to the layouts folder

## Data Collection

In our experiment we compared four different configurations, all combinations of 2 agents and 2 extractors, accross nine different layouts

To simplify data collection, we run all four configurations for a layout at the same time

to run the data collection for a layout run the following command:

`python RunProjectTest.py -l mediumClassic`

The results are stored by layout name and timestamp in the python 'pickle' formant in the 'outputs' directory.

The results we collected for our experiment are located in the 'results' directory

A bash script `runall.sh` will run this data collection for all layouts we analyzed

NOTE: this will take a very long time to run, even with a very good laptop it took around 14 hours to generate data for every layout, you can tweak the parameters at the bottom of RunProjectTest.py to run less training runs to validate that the data collection is working properly.

## Data Analysis

We perform several analysis using the data we collected, all of these are performed by the script `ConvergenceTest.py`

Running this script will load the pickled results from the results directory and perform calculations, display charts, and generate output files.

Some of the analysis of convergence data will be stored in the stat-results folder in a csv format.

The csv output include paired t-tests performed on convergence points and the scores at the convergence points, performed per algorithm and per extractor across all of the layouts.

The data analysis can be performed by running the script without any arguments:

`python ConvergenceTest.py`
