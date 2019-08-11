import util
from sudoku import *
from maps import MapSearchProblem

################ Node structure to use for the search algorithm ################
class Node:
    def __init__(self, state, action, path_cost, parent_node, depth):
        self.state = state
        self.action = action
        self.path_cost = path_cost
        self.parent_node = parent_node
        self.depth = depth

########################## DFS for Sudoku ########################
## Choose some node to expand from the frontier with Stack like implementation
def sudokuDepthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """

    def convertStateToHash(values):
        """ 
        values as a dictionary is not hashable and hence cannot be used directly in the explored set.
        This function changes values dict into a unique hashable string which can be used in the explored set.
        """
        l = list(sorted(values.items()))
        modl = [a+b for (a, b) in l]
        return ''.join(modl)

    ## YOUR CODE HERE
    frontier = util.Stack()
    frontier.push((problem.start_values,((0,'')),0))
    explored_set = set()
    while True:
        l = frontier.pop()
        Curr_state = l[0]
        if problem.isGoalState(Curr_state):
            return Curr_state 
        elif convertStateToHash(Curr_state) in explored_set:
            continue 
        else:
            explored_set.add(convertStateToHash(Curr_state))
            new_frontiers = problem.getSuccessors(Curr_state)
            for new_state in new_frontiers:
                string_rep_new_state = convertStateToHash(new_state[0])
                if not string_rep_new_state in explored_set:
                    frontier.push(new_state)

    util.raiseNotDefined()

######################## A-Star and DFS for Map Problem ########################
## Choose some node to expand from the frontier with priority_queue like implementation

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def heuristic(state, problem):
    # It would take a while for Flat Earther's to get accustomed to this paradigm
    # but hang in there.
    goal = problem.end_node
    pos_state = ((problem.G.node[state]['x'],0,0),(problem.G.node[state]['y'],0,0))
    pos_goal = ((problem.G.node[goal]['x'],0,0),(problem.G.node[goal]['y'],0,0))
    return util.points2distance(pos_state,pos_goal)
   

    util.raiseNotDefined()

def AStar_search(problem, heuristic=nullHeuristic):

    """Search the node that has the lowest combined cost and heuristic first."""

    frontier = util.PriorityQueue()
    startNode = Node(problem.getStartState(),None,0,None,0)
    frontier.push(startNode,heuristic(startNode.state,problem))
    explored_set = set()
    while True:
        Curr_node = frontier.pop()
        Curr_state = Curr_node.state
        if problem.isGoalState(Curr_state):
            path = []
            while Curr_node.depth != 0:
                path.insert(0,Curr_node.state)
                Curr_node = Curr_node.parent_node
            path.insert(0,startNode.state)
            return path
        elif Curr_state in explored_set:
            continue 
        else:
            explored_set.add(Curr_state)
            new_frontiers = problem.getSuccessors(Curr_state)
            for transition in new_frontiers: 
                if not transition[0] in explored_set:
                    len(explored_set)
                    new_node = Node(transition[0],transition[1],transition[2]+Curr_node.path_cost,Curr_node,Curr_node.depth+1)
                    frontier.push(new_node,heuristic(new_node.state,problem)+new_node.path_cost)

                    
    util.raiseNotDefined()