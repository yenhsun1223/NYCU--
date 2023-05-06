# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        return childGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        maxv = ("max", -float("inf"))
        agentIndex = 0
        depth = 0
        for step in gameState.getLegalActions(agentIndex):
            cval = (step, self.minimax(gameState.getNextState(agentIndex, step), (depth + 1)%gameState.getNumAgents(), depth + 1))
            if cval[1] > maxv[1]:
                maxv = cval       
        return maxv[0]        
       
    def minimax(self, gameState, agentIndex, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        maxv = ("max", -float("inf"))
        minv = ("min", float("inf"))    
        for step in gameState.getLegalActions(agentIndex):
            cval = (step, self.minimax(gameState.getNextState(agentIndex, step), (depth + 1)%gameState.getNumAgents(), depth + 1))
            if agentIndex == 0 and cval[1] > maxv[1]:
                maxv = cval
            elif agentIndex >= 1 and cval[1] < minv[1]:
                minv = cval
        if agentIndex == 0: return maxv[1]
        else: return minv[1]      
        # End your code (Part 1)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        maxv = ("max", -float("inf"))
        agentIndex = 0
        depth = 0
        alpha = -float("inf")
        beta = float("inf")
        for step in gameState.getLegalActions(agentIndex):
            cval = (step, self.alphabeta(gameState.getNextState(agentIndex, step), (depth + 1)%gameState.getNumAgents(), depth + 1, alpha, beta))
            if cval[1] > maxv[1]: maxv = cval        
            if maxv[1] > beta: return maxv
            elif maxv[1] > alpha: alpha = maxv[1]
        return maxv[0]        

    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        maxv = ("max", -float("inf"))
        minv = ("min", float("inf"))    
        for step in gameState.getLegalActions(agentIndex):
            cval = (step, self.alphabeta(gameState.getNextState(agentIndex, step), (depth + 1)%gameState.getNumAgents(), depth + 1, alpha, beta))
            if agentIndex == 0:
                if cval[1] > maxv[1]: maxv = cval               
                if maxv[1] > beta: return maxv[1]                
                elif maxv[1] > alpha: alpha = maxv[1]
            else:
                if cval[1] < minv[1]: minv = cval                  
                if minv[1] < alpha: return minv[1]                  
                elif minv[1] < beta: beta = minv[1]                             
        if agentIndex == 0: return maxv[1]
        else: return minv[1] 
        # End your code (Part 2)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        return self.expectimax(gameState, 0, self.depth * gameState.getNumAgents(), "expectimaxagent")[0]

    def expectimax(self, gameState, agentIndex, depth, action):

        if depth == 0 or gameState.isLose() or gameState.isWin():
            return (action, self.evaluationFunction(gameState))

        maxv = ("max", -float('inf'))
        expv = 0.0
        for step in gameState.getLegalActions(agentIndex):     
          if agentIndex == 0:     
            cval = self.expectimax(gameState.getNextState(agentIndex, step), (agentIndex + 1) % gameState.getNumAgents(), depth - 1, step)
            if cval[1] > maxv[1]:
                maxv = cval
          else:
            v = self.expectimax(gameState.getNextState(agentIndex, step), (agentIndex + 1) % gameState.getNumAgents(), depth - 1, step)
            expv += v[1] / len(gameState.getLegalActions(agentIndex))           
        if agentIndex == 0: return maxv    
        else: return (action, expv)             


        # End your code (Part 3)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme evaluation function
    """
    # Begin your code (Part 4)
    distance = []
    min_distance = 0
    t = 0
    for i in currentGameState.getFood().asList():
        distance.append(manhattanDistance(i, currentGameState.getPacmanPosition()))
    if len(distance):
        min_distance = min(distance)
    return currentGameState.getScore() - min_distance
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
