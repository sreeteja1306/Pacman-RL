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

def setAtRandomPosition(grid, symbol):
    while True:
        r_pos = random.randint(0,len(grid)-1)
        c_pos = random.randint(0, len(grid[0])-1)
        if grid[r_pos][c_pos] == '.':
            grid[r_pos][c_pos] = symbol
            return

class MazeParams:
    def __init__(self, name, height, width):
        self.name: str = name
        self.height: int = height
        self.width: int = width
        self.wallChance: int = 0
        self.ghosts: int = 2
        self.capsules: int = 2

def GenerateLayouts(params_list: list[MazeParams]):
    pacman = 'P'
    ghost = 'G'
    capsule = 'o'
    wall = '%'
    pellet = '.'
    unkwn = '?'

    height = params_list[0].height
    width = params_list[0].width

    # enforce odd height/width
    height = (height//2)*2 + 1
    width = (width//2)*2 + 1

    # create grid
    grid = [[unkwn for _ in range(width)] for _ in range(height)]

    # add border walls
    for idx in range(len(grid)):
        for jdx in range(len(grid[idx])):
            if idx == 0 or idx == (height-1):
                grid[idx][jdx] = wall
            if jdx == 0 or jdx == (width-1):
                grid[idx][jdx] = wall

    # starting cell 
    start_x = random.randint(2, width - 3)
    start_y = random.randint(2, height - 3)
    grid[start_y][start_x] = pellet

    stack = [(start_y, start_x)]

    # functions for neighbor calculations
    def N(y, x): return (y+2, x) if y+2 < height and grid[y+2][x] == unkwn else None
    def E(y, x): return (y, x+2) if x+2 < width and grid[y][x+2]  == unkwn else None
    def S(y, x): return (y-2, x) if y-2 > 0 and grid[y-2][x] == unkwn else None
    def W(y, x): return (y, x-2) if x-2 > 0 and grid[y][x-2] == unkwn else None

    # growing tree maze generation
    while stack:
        cpos = stack[-1]
        cy, cx = cpos

        neighbors = [N(*cpos), E(*cpos), S(*cpos), W(*cpos)]
        neighbors = [n for n in neighbors if n is not None]

        if not neighbors: 
            stack = list(filter(lambda p: p != cpos, stack))
            continue

        n2pos = random.choice(neighbors)
        stack.append(n2pos)

        n2_y, n2_x = n2pos
        grid[n2_y][n2_x] = pellet

        my, mx = (stat.mean([cy, n2_y]), stat.mean([cx, n2_x]))
        grid[my][mx] = pellet

    # fill in walls
    for r, _ in enumerate(grid):
        for c, _ in enumerate(grid[r]):
            if grid[r][c] == unkwn: 
                grid[r][c] = wall
    
    # check that the algorithm didn't double wall anywhere, just toss it and retry if it did
    # not really necessary but it looks nicer without double borders
    col1 = [grid[i][1] for i in range(height)]
    col2 = [grid[i][-2] for i in range(height)]
    retry = False
    if grid[1] == [wall] * width: retry = True
    if grid[-2] == [wall] * width: retry = True
    if col1 == [wall] * height: retry = True
    if col2 == [wall] * height: retry = True
    if retry:
        return GenerateLayouts(params_list)

    entity_layouts = []
    used_ghost_capsule = []
    for lparams in params_list:
        if (lparams.ghosts, lparams.capsules) in used_ghost_capsule:
            continue
        used_ghost_capsule.append((lparams.ghosts, lparams.capsules))

        temp_layout = deepcopy(grid)
        random.seed(0)
        # set positions for non-wall elements
        setAtRandomPosition(temp_layout, pacman)
        for _ in range(lparams.ghosts):
            setAtRandomPosition(temp_layout, ghost)
        for _ in range(lparams.capsules):
            setAtRandomPosition(temp_layout, capsule)
        entity_layouts.append(temp_layout)

    print(len(entity_layouts))

    final_layouts = []
    used_wallchances = []
    for lparams in params_list:
        if lparams.wallChance in used_wallchances:
            continue
        used_wallchances.append(lparams.wallChance)
        for layout in entity_layouts:
            random.seed(0)
            # randomly remove walls to create a more open environment
            temp_grid = deepcopy(layout)
            for r, _ in enumerate(temp_grid):
                for c, _ in enumerate(temp_grid[r]):
                    # needs to make sure not to remove border
                    if r == 0 or r == height-1:
                        continue
                    if c == 0 or c == width-1:
                        continue
                    if temp_grid[r][c] == wall and util.flipCoin(lparams.wallChance):
                        temp_grid[r][c] = pellet
            final_layouts.append(temp_grid)

    def output_layout(grid,lparams):
        layout_lines = ["".join(c) for c in grid]
        layout_string = "\n".join(layout_lines)
        output_file = f"./layouts/{lparams.name}-{int(time.time())}.lay"
        with open(output_file, 'w') as f:
            f.write(layout_string)
        return layout_string
    
    layout_objs = []
    for idx, lparams in enumerate(params_list):
        layout_string = output_layout(final_layouts[idx], lparams)
        layout_objs += [la.Layout([line.strip() for line in layout_string.split('\n')], lparams.name)]
    return layout_objs



if __name__ == "__main__":
    sparse_simple = MazeParams('sparse_simple',15,15)
    sparse_simple.wallChance =.2
    sparse_simple.ghosts = 3
    sparse_simple.capsules = 2

    dense_simple = MazeParams('dense_simple',15,15)
    dense_simple.wallChance =.4
    dense_simple.ghosts = 3
    dense_simple.capsules = 2

    sparse_complex = MazeParams('sparse_complex',15,15)
    sparse_complex.wallChance =.4
    sparse_complex.ghosts = 4
    sparse_complex.capsules = 7

    dense_complex = MazeParams('dense_complex',15,15)
    dense_complex.wallChance =.2
    dense_complex.ghosts = 4
    dense_complex.capsules = 7

    params = [sparse_simple, dense_complex, dense_simple, sparse_complex]

    layouts = GenerateLayouts(params)

    for idx, layout in enumerate(layouts):
        ghost = [ghostType(i+1) for i in range(params[idx].ghosts)]
        displayLayout(layout, ghost)
    
    medium_complex = MazeParams('medium_complex',11,20)
    medium_complex.wallChance = .2
    medium_complex.ghosts = 4
    medium_complex.capsules = 7

    params = [medium_complex]

    layouts = GenerateLayouts(params)

    for idx, layout in enumerate(layouts):
        ghost = [ghostType(i+1) for i in range(params[idx].ghosts)]
        displayLayout(layout, ghost)