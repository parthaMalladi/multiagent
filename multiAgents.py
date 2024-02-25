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
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodCoords = newFood.asList()
        distanceToFood = []
        
        for coord in foodCoords:
            distanceToFood.append(manhattanDistance(newPos, coord))
        
        closestDistanceToFood = -1
        if (len(distanceToFood) > 0):
            closestDistanceToFood = min(distanceToFood)
        
        ghostDistance = 1
        ghostCoords = successorGameState.getGhostPositions()
        ghostProximity = 0
        
        for ghost in ghostCoords:
            dist = manhattanDistance(newPos, ghost)
            ghostDistance += dist
            if dist < 2:
                ghostProximity += 1
        
        return successorGameState.getScore() + (1/closestDistanceToFood) - (1/ghostDistance) - ghostProximity
    

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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        ghosts = gameState.getNumAgents() - 1
        return self.maximize(gameState, 1, ghosts)
        
    def maximize(self, gameState, depth, numGhosts):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        maxValue = float('-inf')
        bestAction = Directions.STOP
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            tempValue = self.minimize(successor, depth, 1, numGhosts)
            if tempValue > maxValue:
                maxValue = tempValue
                bestAction = action
        
        if depth > 1:
            return maxValue

        return bestAction

    def minimize(self, gameState, depth, agentIndex, numGhosts):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        minValue = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == numGhosts:
                if depth < self.depth:
                    minValue = min(minValue, self.maximize(successor, depth + 1, numGhosts))
                else:
                    minValue = min(minValue, self.evaluationFunction(successor))
            else:
                minValue = min(minValue, self.minimize(successor, depth, agentIndex + 1, numGhosts))
        
        return minValue

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numGhosts = gameState.getNumAgents() - 1
        return self.maximize(gameState, 1, numGhosts, float('-inf'), float('inf'))
    
    def maximize(self, gameState, depth, numGhosts, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        maxValue = float('-inf')
        bestAction = Directions.STOP
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            tempValue = self.minimize(successor, depth, 1, numGhosts, alpha, beta)
            if maxValue < tempValue:
                maxValue = tempValue
                bestAction = action
            
            if maxValue > beta:
                return maxValue
            alpha = max(alpha, maxValue)
            
        if depth > 1:
            return maxValue

        return bestAction

    def minimize(self, gameState, depth, agentIndex, numGhosts, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        minValue = float('inf')
        
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == numGhosts:
                if depth < self.depth:
                    minValue = min(minValue, self.maximize(successor, depth + 1, numGhosts, alpha, beta))
                else:
                    minValue = min(minValue, self.evaluationFunction(successor))
            else:
                minValue = min(minValue, self.minimize(successor, depth, agentIndex + 1, numGhosts, alpha, beta))
            
            if minValue < alpha:
                return minValue
            
            beta = min(beta, minValue)
        
        return minValue      

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
        numGhosts = gameState.getNumAgents() - 1
        return self.maximize(gameState, 1, numGhosts)
    
    def maximize(self, gameState, depth, numGhosts):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        maxValue = float('-inf')
        bestAction = Directions.STOP
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            tempValue = self.getExpectValue(successor, depth, 1, numGhosts)
            
            if maxValue < tempValue:
                maxValue = tempValue
                bestAction = action
                
        if depth > 1:
            return maxValue
        
        return bestAction
    
    def getExpectValue(self, gameState, depth, agentIndex, numGhosts):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        expectValue = 0
        legalActions = gameState.getLegalActions(agentIndex)
        successor_probability = 1.0 / len(legalActions)
        
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == numGhosts:
                if depth < self.depth:
                    expectValue += successor_probability * self.maximize(successor, depth + 1, numGhosts)
                else:
                    expectValue += successor_probability * self.evaluationFunction(successor)
            else:
                expectValue += successor_probability * self.getExpectValue(successor, depth, agentIndex + 1, numGhosts)
        
        return expectValue

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    I am just subtracting various values from the current game state's score. I reduce the score by a lot if there
    are many power capsules on the board because I want pacman to be prioritize scaring the ghosts. I also reduce the
    the score if there is a lot of food remaining on the board so that pacman keeps moving. I subtract the minimum distance
    between pacman and food pellet because it makes pacman get a worse score if he is farther away from food, so
    he will prioritize being near food pellets. I am also tracking the ghost positions and which ghosts are scared.
    Pacman has a lower penalty if he is farther away from a ghost because I use reciprocal of the manhattan distance
    and subtract it from the game state's score. I also penalize pacman for being far away from a scared ghost, which
    will make pacman prioritize hunting ghosts when they are scared.
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    pacman_position = currentGameState.getPacmanPosition()
    food_positions = currentGameState.getFood().asList()
    capsules_positions = currentGameState.getCapsules()
    ghost_states = currentGameState.getGhostStates()
    remaining_food = len(food_positions)
    remaining_capsules = len(capsules_positions)
    scared_ghosts = []
    enemy_ghosts = []
    enemy_ghost_positions = []
    scared_ghosts_positions = []

    closest_food = float('inf')
    closest_enemy_ghost = float('inf')
    closest_scared_ghost = float('inf')

    distance_from_food = []
    for food in food_positions:
        distance_from_food.append(manhattanDistance(pacman_position, food))
        
    if len(distance_from_food) > 0:
        closest_food = min(distance_from_food)
        score -= closest_food

    for ghost in ghost_states:
        if ghost.scaredTimer > 0:
            enemy_ghosts.append(ghost)
        else:
            scared_ghosts.append(ghost)

    for enemy_ghost in enemy_ghosts:
        enemy_ghost_positions.append(enemy_ghost.getPosition())

    if len(enemy_ghost_positions) > 0:
        distance_from_enemy_ghost = []
        for enemy_ghost_position in enemy_ghost_positions:
            distance_from_enemy_ghost.append(manhattanDistance(pacman_position, enemy_ghost_position))
            
        closest_enemy_ghost = min(distance_from_enemy_ghost)
        score -= 3.0 * (1 / closest_enemy_ghost)

    for scared_ghost in scared_ghosts:
        scared_ghosts_positions.append(scared_ghost.getPosition())

    if len(scared_ghosts_positions) > 0:
        distance_from_scared_ghost = []
        for scared_ghost_position in scared_ghosts_positions:
            distance_from_scared_ghost.append(manhattanDistance(pacman_position, scared_ghost_position))
            
        closest_scared_ghost = min(distance_from_scared_ghost)
        score -= 3.0 * closest_scared_ghost

    score -= 20.0 * remaining_capsules
    score -= 3.0 * remaining_food
    return score

# Abbreviation
better = betterEvaluationFunction
