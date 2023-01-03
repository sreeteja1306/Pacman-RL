from pacman import GameState
from pacman import loadAgent
from copy import deepcopy
import statistics as stat
import graphicsDisplay
import layout as la
import random
import util
import time

display = graphicsDisplay.PacmanGraphics(1.0, frameTime=30)
initState = GameState()
ghostType = loadAgent("RandomGhost", True)

def displayLayout(layout, ghostAgents):
    initState.initialize(layout, len(ghostAgents))
    display.initialize(initState.data)
    time.sleep(5)



import glob
import sys
import os
if __name__ == "__main__":
    if len(sys.argv) == 2:
        layout_name = './layouts/' + sys.argv[1]
        if not os.path.exists(layout_name):
            raise Exception("File %s does not exist" % sys.argv[1])
    else:
        layout_files = glob.glob(f'./layouts/*')
        layout_name = max(layout_files, key=os.path.getctime)

    print("Loading layout file %s" % layout_name) 

    def getlayout():
        output_file = f"{layout_name}"
        with open(output_file, 'r') as f:
            layout_data = f.read()
        return layout_data

    raw_layout = getlayout()
    ghost_ct = len([c == 'G' for c in raw_layout])
    ghost = [ghostType(i+1) for i in range(ghost_ct)]
    layout = la.Layout([line.strip() for line in getlayout().split('\n')], layout_name)
    displayLayout(layout, ghost)