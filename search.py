# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class Node:
    def __init__(self, parent, coordinates, direction, cost):
        self.parent = parent
        self.state = coordinates
        self.action = direction
        self.cost = cost

    def __eq__(self, other):
        return self.state == other.state

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    startNode = problem.getStartState()
    if problem.isGoalState(startNode):
        return []

    s = util.Stack()
    s.push((startNode, []))
    visN = []

    while not s.isEmpty():
        tempN, tempA = s.pop()

        if problem.isGoalState(tempN):
            return tempA

        if tempN not in visN:
            visN.append(tempN)
            for node, action, cost in problem.getSuccessors(tempN):
                s.push((node, tempA + [action]))

    """start = problem.getStartState()
    solution = list()
    explored = list()
    frontier = util.Stack()

    frontier.push(Node(None, start, None, 0))

    while not frontier.isEmpty():
        current = frontier.pop()

        if problem.isGoalState(current.state):

            while current.parent:
                solution.append(current.action)
                current = current.parent

            solution.reverse()
            return solution

        explored.append(current.state)
        successors = problem.getSuccessors(current.state)

        for node, action, cost in successors:
            node = Node(current, node, action, 0)
            if node.state not in explored:
                frontier.push(node)

    util.raiseNotDefined()

    return None"""


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    startN = problem.getStartState();
    if problem.isGoalState(startN):
        return []

    queue = util.Queue()
    queue.push((startN, []))
    visN = []
    while not queue.isEmpty():
        tempN, tempA = queue.pop()
        if problem.isGoalState(tempN):
            return tempA
        if tempN not in visN:
            visN.append(tempN)

            for node, action, cost in problem.getSuccessors(tempN):
                if node not in queue.list:
                    queue.push((node, tempA + [action]))

    util.raiseNotDefined()
    """start = problem.getStartState()
    solution = list()
    explored = list()
    frontier = util.Queue()

    frontier.push(Node(None, start, None, 0))

    while not frontier.isEmpty():
        current = frontier.pop()

        if problem.isGoalState(current.state):

            while current.parent:
                solution.append(current.action)
                current = current.parent

            solution.reverse()
            return solution

        explored.append(current.state)
        successors = problem.getSuccessors(current.state)

        for node, action, cost in successors:
            node = Node(current, node, action, 0)
            if (node.state not in explored) and (node not in frontier.list):
                frontier.push(node)"""

    util.raiseNotDefined()

    return None

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    startNode = problem.getStartState()
    if problem.isGoalState(startNode):
        return []
    pQ = util.PriorityQueue()
    pQ.push((startNode, [], 0), 0) #the priority is given by the cost of the "route"
    visited = []

    while not pQ.isEmpty():
        temp_node, temp_actions, temp_cost = pQ.pop()
        if problem.isGoalState(temp_node):
            return temp_actions
        if temp_node not in visited:
            visited.append(temp_node)

            for node, action, cost in problem.getSuccessors(temp_node):
                newCost = cost + temp_cost
                newActions = temp_actions + [action]
                pQ.push((node, newActions, newCost), newCost)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    startNode = problem.getStartState()
    if problem.isGoalState(startNode):
        return []
    pQ = util.PriorityQueue()
    pQ.push((startNode, [], 0), 0)  # the priority is given by the cost of the "route"
    visited = []

    while not pQ.isEmpty():
        temp_node, temp_actions, temp_cost= pQ.pop()
        if problem.isGoalState(temp_node):
            return temp_actions
        if temp_node not in visited:
            visited.append(temp_node)
            for node, action, cost in problem.getSuccessors(temp_node):
                newCost = cost + temp_cost
                newActions = temp_actions + [action]
                costOfHeur = newCost + heuristic(node, problem)
                pQ.push((node, newActions, newCost), costOfHeur)

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
