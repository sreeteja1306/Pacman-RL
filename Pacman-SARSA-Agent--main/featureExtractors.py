# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

def closestFoodCapsules(pos, food, capsules, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y] or (pos_x, pos_y) in capsules:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None


def closestGhost(pos, ghosts, walls,Max_Depth = None):
    """
    closestGhost -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    legal_ghosts = []
    for ghost in ghosts:
        legal_ghosts += [(ghost[0]//1, ghost[1]//1)]
    if len(legal_ghosts) == 0:
        return None
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded or (Max_Depth is not None and dist > Max_Depth):
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if (pos_x, pos_y) in legal_ghosts:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no Ghost found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

class SpookedExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether an unscared ghost is one step away
    - whether a scared ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        capsules = state.getCapsules()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        ghost_states = state.getGhostStates()
        ghosts_scared = [s.getPosition() for s in ghost_states if s.scaredTimer > 0]

        def getNeighbors2(g, walls):
            neighbors_two_step = set()
            for neighbor in Actions.getLegalNeighbors(g, walls):
                neighbors_two_step.add(neighbor)
                for neighbor2 in Actions.getLegalNeighbors(neighbor, walls):
                    neighbors_two_step.add(neighbor2)
            return list(neighbors_two_step)

        # count the number of scared ghosts 1-step away
        # features["#-of-scared-ghosts-2-step-away"] = sum((next_x, next_y) in getNeighbors2(g, walls) for g in ghosts_scared)

        # # count the number of unscared ghosts 1-step away
        # features["#-of-unscared-ghosts-2-step-away"] = sum((next_x, next_y) in getNeighbors2(g, walls) for g in ghosts if g not in ghosts_scared)

        # # count the number of scared ghosts 1-step away
        # features["#-of-scared-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts_scared)

        # # count the number of unscared ghosts 1-step away
        # features["#-of-unscared-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts if g not in ghosts_scared)

        not_scared = list(set(ghosts) - set(ghosts_scared))
        dist_unscared = closestGhost((next_x, next_y), not_scared, walls,2)
        dist_scared = closestGhost((next_x, next_y), ghosts_scared, walls)
        if dist_scared is not None:
            features["closest-ghost-scared"] = 1/pow(max(.25, float(dist_scared)), 3)
        if dist_unscared is not None:
            features["closest-ghost-unscared"] = 1/pow(max(.25, float(dist_unscared)), 3)

        dist = closestFoodCapsules((next_x, next_y), food, capsules, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        # if there is no danger of ghosts then add the food feature
        if dist_unscared is None and food[next_x][next_y]:
            features["eats-food"] = 1.0
        
        # if there is no danger of ghosts then add the capsule feature
        if dist_unscared is None and (next_x, next_y) in capsules:
            features["eats-capsule"] = 1.0

        # from IPython import embed; embed()
        # exit()
        features.divideAll(10.0)
        return features
