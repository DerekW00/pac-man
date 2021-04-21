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
#import numpy as np

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

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    #def calculate_all_distance(new_postition, relevant_stuff):
        #eturn util.manhattanDistance(new_postition, relevant_stuff )

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

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
        #print(newPos, newFood, newGhostStates, newScaredTimes)
        minimum_scared_time = min(newScaredTimes)
        newFood = newFood.asList()
        gPosition = [(state.getPosition()[0], state.getPosition()[1]) for state in newGhostStates]
        if not minimum_scared_time and (newPos in gPosition):
            return -1.0

        if newPos in currentGameState.getFood().asList():
            return 1.0

        min_F_dis = sorted(newFood, key= lambda fD: util.manhattanDistance(fD, newPos))
        min_g_dis = sorted(gPosition, key=lambda gD: util.manhattanDistance(gD, newPos))

        food_dis = lambda f: util.manhattanDistance(f, newPos)
        ghost_dis = lambda g: util.manhattanDistance(g, newPos)

        return 1.0 / food_dis(min_F_dis[0]) - 1.0 / ghost_dis(min_g_dis[0])

    

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
    Your minimax agent (question 2)
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

        """ Inspired by code implemented in AIMA"""
        total_agents = gameState.getNumAgents() #iterate total_agents -1 times
        g_ind = [g for g in range(1, total_agents)]
        depth_initial = self.depth
        #initial_state = gameState

        def termination (state, depth):
            return depth == depth_initial or state.isWin() or state.isLose()
        
        
        def helper_min_evaluator(state, depth, ghost_index):
            if termination(state, depth):
                return self.evaluationFunction(state) 
            #value = np.inf
            value = 1881888188188188818881881888

            for move in state.getLegalActions(ghost_index):
                if ghost_index == g_ind[-1]:
                    value = min(value, helper_max_evaluator(state.getNextState(ghost_index, move), depth+1))
                else:
                    value = min (value, helper_min_evaluator(state.getNextState(ghost_index, move), depth, ghost_index+1))
            return value


        def helper_max_evaluator(state, depth):
            if termination(state, depth):
                return self.evaluationFunction(state)
            #value = -np.inf
            value = -18881881881881881881881881881888188888888888

            for moves in state.getLegalActions(0):
                #if ghost_index == g_ind[-1]:
                    #value = max(value, helper_min_evaluator(state.getNextState(ghost_index, moves), depth+1))
                #else:
                value = max(value, helper_min_evaluator(state.getNextState(0, moves), depth, 1))
            return value


        #while depth > 0:

        all_action = []
        
        for i in gameState.getLegalActions(0):
            all_action.append((i, helper_min_evaluator(gameState.getNextState(0, i), 0, 1)))
        all_action.sort(key=lambda s: s[1])
        return all_action[-1][0]
        
            

       
        



        
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        total_agents = gameState.getNumAgents() #iterate total_agents -1 times
        g_ind = [g for g in range(1, total_agents)]
        depth_initial = self.depth

        def termination (state, depth):
            return depth == depth_initial or state.isWin() or state.isLose()

    # Functions used by alpha_beta
        def helper_max_value(state, alpha, beta, depth):
            if termination(state, depth):
                return self.evaluationFunction(state)
            value = -188188188188188188818888888881888888
            for a in state.getLegalActions(0):
                value = max(value, helper_min_value(state.getNextState(0, a), alpha, beta, depth, 1))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value

        def helper_min_value(state, alpha, beta, depth, ghost_index):
            if termination(state, depth):
                return self.evaluationFunction(state)
            value = 1888888888888888888888888
            for a in state.getLegalActions(ghost_index):
                if ghost_index == g_ind[-1]:
                    value = min(value, helper_max_value(state.getNextState(ghost_index, a), alpha, beta, depth+1))
                else:
                    value= min(value, helper_min_value(state.getNextState(ghost_index, a), alpha, beta, depth, ghost_index+1))
                    
                    
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

    # Body of alpha_beta_search:
        def running(state):
            v = -18888888888888888888
            act = None
            alpha = -18888888888888888888
            beta = 188888888888888888

            for action in state.getLegalActions(0):  # maximizing
                middle = helper_min_value(gameState.getNextState(0, action), alpha, beta, 0, 1)

                if v < middle:
                    v = middle
                    act = action

                if v > beta: 
                    return v
                alpha = max(alpha, middle)

            return act
        return running(gameState)

        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        total_agents = gameState.getNumAgents() #iterate total_agents -1 times
        g_ind = [g for g in range(1, total_agents)]
        depth_initial = self.depth
        #initial_state = gameState

        def termination (state, depth):
            return depth == depth_initial or state.isWin() or state.isLose()
        
        def expect(state, depth, ghost):
            if termination(state, depth):
                return self. evaluationFunction(state)
            value = 0
            probability = 1/len(state.getLegalActions(ghost))

            for a in state.getLegalActions(ghost):
                if ghost == g_ind[-1]:
                    value += probability*helper_max_evaluator(state.getNextState(ghost, a), depth+1)
                else:
                    value += probability*expect(state.getNextState(ghost, a), depth, ghost+1)

            return value

        def helper_min_evaluator(state, depth, ghost_index):
            if termination(state, depth):
                return self.evaluationFunction(state) 
            #value = np.inf
            
            
            value = 1881888188188188818881881888

            for move in state.getLegalActions(ghost_index):
                if ghost_index == g_ind[-1]:
                    value = min(value, helper_max_evaluator(state.getNextState(ghost_index, move), depth+1))
                else:
                    value = min (value, helper_min_evaluator(state.getNextState(ghost_index, move), depth, ghost_index+1))
            return value


        def helper_max_evaluator(state, depth):
            if termination(state, depth):
                return self.evaluationFunction(state)
            #value = -np.inf
            value = -18881881881881881881881881881888188888888888

            for moves in state.getLegalActions(0):
                #if ghost_index == g_ind[-1]:
                    #value = max(value, helper_min_evaluator(state.getNextState(ghost_index, moves), depth+1))
                #else:
                value = max(value, helper_min_evaluator(state.getNextState(0, moves), depth, 1))
            return value


        #while depth > 0:

        all_action = []
        
        for i in gameState.getLegalActions(0):
            all_action.append((i, expect(gameState.getNextState(0, i), 0, 1)))
        all_action.sort(key=lambda s: s[1])
        return all_action[-1][0]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: Adjustment from the alg. in question 1. Added more features for heuristics.
    """
   #childGameState = currentGameState.getPacmanNextState(action)
    
    
    food_count = 0
    power_count = 0
    ghost_count = 0

    Pos, Food = currentGameState.getPacmanPosition(), currentGameState.getFood()
    currGhostPos, currPower =  currentGameState.getGhostPositions(), currentGameState.getCapsules()
    
    
    newFood = Food.asList()

    min_Food = float(min([Food.width + Food.height] + [util.manhattanDistance(Pos, i) for i in newFood]))
    if len(newFood) == 0:
        food_count = 1
    else:
        food_count = 1/min_Food
    
    
    
    min_Ghost = float(min([util.manhattanDistance(Pos, i) for i in currGhostPos]))
    
    
    if min_Ghost < 1:
        ghost_count = -99
    else:
        ghost_count = 1/min_Ghost

    coeff_g = 1
    


    closestPow = float(min([len(currPower)] + [util.manhattanDistance(powerPos, Pos) for powerPos in currPower]))
    power_count = 1 if len(currPower)==0 else 1/closestPow


    scaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]
    isScared = True if max(scaredTimes)!=0 else False

    if isScared and min_Ghost < max(scaredTimes):
        coeff_g, ghost_count = 100, abs(ghost_count)

    return food_count + coeff_g*ghost_count + power_count + currentGameState.getScore()

   # gPosition = [(state.getPosition()[0], state.getPosition()[1]) for state in newGhostStates]
        #if not minimum_scared_time and (newPos in gPosition):
            #return -1.0

        #if newPos in currentGameState.getFood().asList():
            #return 1.0

        #min_F_dis = sorted(newFood, key= lambda fD: util.manhattanDistance(fD, newPos))
        #min_g_dis = sorted(gPosition, key=lambda gD: util.manhattanDistance(gD, newPos))

        #food_dis = lambda f: util.manhattanDistance(f, newPos)
        #ghost_dis = lambda g: util.manhattanDistance(g, newPos)

        #return 1.0 / food_dis(min_F_dis[0]) - 1.0 / ghost_dis(min_g_dis[0])

better = betterEvaluationFunction
