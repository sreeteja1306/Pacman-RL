from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class SARSALearningAgent(ReinforcementAgent):
    """
          SARSA-Learning Agent

          Functions you should fill in:
            - computeValueFromQValues
            - computeActionFromQValues
            - getQValue
            - getAction
            - update

          Instance variables you have access to
            - self.epsilon (exploration prob)
            - self.alpha (learning rate)
            - self.discount (discount rate)
            - self.lamda (trace decay rate)

          Functions you should use
            - self.getLegalActions(state)
              which returns legal actions for a state
        """

    def __init__(self, lamda=0.3,**args):
        "You can initialize Q-values here..."
        self.lamda = float(lamda)
        self.trace = util.Counter()
        ReinforcementAgent.__init__(self, **args)
        self.Qvalues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.Qvalues[(state,action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        if len(self.getLegalActions(state)) == 0:
            return 0
        value = []
        for action in self.getLegalActions(state):
            value.append(self.getQValue(state, action))
        return max(value)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = self.getLegalActions(state)
        max_val = None 
        max_act = None
        for a in actions:
          qv = self.getQValue(state, a)
          if max_val is None: max_val = qv; max_act = a
          if max_val == qv: max_act = random.choice([max_act, a])
          if max_val < qv: max_val = qv; max_act = a
        return max_act

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

class PacmanSARSAAgent(SARSALearningAgent):
    "Exactly the same as SARSALearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.001, numTraining=0,extractor="IdentityExtractor", **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        extractor - feature extractor function
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.featExtractor = util.lookup(extractor, globals())()
        self.weights = util.Counter()
        self.QOld = 0
        self.action = 0
        SARSALearningAgent.__init__(self, **args)

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        QValue = 0
        for feature in features:
            QValue += self.weights[feature]*features[feature]
        return QValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        NextAction = 0
        if self.isInTraining():
            featuresCurrent = self.featExtractor.getFeatures(state, action)
            QValueCurrent = self.getQValue(state, action)
            delta = reward - QValueCurrent
            if self.getLegalActions(nextState):
                NextAction = SARSALearningAgent.getAction(self, nextState)
                QValueNext = self.getQValue(nextState, NextAction)
                delta += self.discount*QValueNext
            traceFactor = 0
            for feature in featuresCurrent:
                traceFactor += self.trace[feature]*featuresCurrent[feature]
            traceFactor = self.discount * self.lamda * self.alpha * traceFactor
            for feature in featuresCurrent:
                self.trace[feature] = self.discount * self.lamda*self.trace[feature] \
                                    + (1-traceFactor)*featuresCurrent[feature]
                self.weights[feature] = self.weights[feature] \
                                        +  self.alpha*(delta+QValueCurrent-self.QOld)*self.trace[feature] \
                                        - self.alpha * (QValueCurrent - self.QOld) * featuresCurrent[feature]
            if self.getLegalActions(nextState):
                self.QOld = QValueNext
        self.action = NextAction


    def final(self, state):
        "Called at the end of each game."
        self.trace = util.Counter()
        self.QOld = 0
        self.action = 0
        SARSALearningAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            pass

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        if self.action == 0:
            action = SARSALearningAgent.getAction(self,state)
        else:
            action = self.action
        self.doAction(state,action)
        return action

