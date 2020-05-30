"""File: astar.py

Authors: MBWhitestone, csirika & stefanklut
UvA Computational Social Choice Project.

This file contains an implementation of astar in python based on:
- UvA Datastructuren & Algoritmen 2018
- https://en.wikipedia.org/wiki/A*_search_algorithm
- https://www.redblobgames.com/pathfinding/a-star/implementation.html

May 2020
"""

from queue import PriorityQueue
import numpy as np


def a_star(start, grid, winner, total_units, brute_force=False):
    """Returns optimal thing.

    start: ([[d1], [d2], ..., [dn]], [u1, ..., un])
     tuple of list of emtpy districts and list of units to assign to districts.
    grid: [[n11, ...n1j], ... [ni1, ... nij]]
     array of nodes
    """
    assert winner in [0, 1]
    district_size = total_units / len(start[0])

    # List of visited nodes
    visited = set()
    # key: node1, value: node2 with lowest cost to node1
    came_from = {}
    # Cost from start to key to estimated goal
    total_cost = {hash_state(start): heuristic(start, total_units, winner,
                                               brute_force)}
    # Cost from start to key
    real_cost = {hash_state(start): 0}

    # PriorityQueue in order of cost from start to goal via the second element
    to_visit = PriorityQueue()
    to_visit.put((total_cost[hash_state(start)],
                 (total_cost[hash_state(start)], start)))

    while not to_visit.empty():
        # Get node with lowest current cost
        _, (_, current) = to_visit.get()
        # current = states[current_index]

        # Check if we are done
        if goal(current):
            return current

        # Make a move
        visited.add(hash_state(current))
        moves = move(current, grid, visited, district_size)
        for mv in moves:
            hmv = hash_state(mv)
            if hmv not in visited:
                # Predicted cost = cost from start to current and from current
                # to the neighbour
                pred_cost = (real_cost[hash_state(current)] +
                             cost(current, mv, grid, winner, district_size))
                # Update queue and lists if needed
                if hmv not in real_cost or pred_cost < real_cost[hmv]:
                    came_from[hmv] = current
                    real_cost[hmv] = pred_cost
                    total_cost[hmv] = real_cost[hmv] + \
                                      heuristic(mv, total_units, winner,
                                                brute_force)

                    # print('Total', total_cost[hmv])
                    to_visit.put((total_cost[hmv], (total_cost[hmv], mv)))

    # No solution found
    return []

def hash_state(state):
    """Hashable version of state."""
    return frozenset(frozenset(s) for s in state[0])

def n_districts(districts, winner, d_size):
    """Get number of lost districts based on how full they are."""
    losses = []
    for d in districts:
        if d:
            weight = len(d) / d_size
            winner_score = np.round(np.mean([unit.voters for unit in d]))
            if winner:
                winner_score = 1 - winner_score
            losses.append(weight * winner_score)

    return sum(losses)

def cost(state, move, grid, winner, d_size):
    """The (expected) number of districts lost to the other party."""
    return max(0, n_districts(state[0], winner, d_size) -
               n_districts(move[0], winner, d_size))

def borders(district, unit):
    """Check if a unit borders a district."""
    if district == []:
        return True
    neighbour_coords = [(unit.x+i, unit.y+j) for i in [1, 0, -1]
                        for j in [1, 0, -1] if bool(i) ^ bool(j)]
    district_coords = [(d_unit.x, d_unit.y) for d_unit in district]
    return bool([i for i in neighbour_coords if i in district_coords])

def move(state, grid, visited, district_size):
    """assigning one more cell to a district
    (in a way that does not violate you constraints).
    """
    moves = []

    districts, rest = state
    for i, unit in enumerate(rest):
        for j in range(len(districts)):
            district = districts[j]
            remaining = districts[:j] + districts[j+1:]
            if borders(district, unit) and len(district) < district_size:
                possible = (remaining + ([district + [unit]]),
                            rest[:i] + rest[i+1:])
                if hash_state(possible) not in visited and possible not in moves:
                    moves.append(possible)

    # Check dead ends.
    for i, d in enumerate(hash_state(state)):
        if len(d) < district_size and d:
            for new_state in [hash_state(m) for m in moves]:
                if d not in new_state:
                    break
            else:
                return []

    return moves

def goal(state):
    """Returns whether all node are in a district."""
    return state[1] == []

def heuristic(state, total_units, winner, brute_force=False):
    """The cost you would get if there were no geometrical constraints for the
       remaining cells.
    """
    # Heuristic should not dominate!
    if goal(state) or brute_force:
        return 0

    votes = np.array([unit.voters for unit in state[1]]).flatten()
    lose_ratio = 1 - np.mean(votes) if winner else np.mean(votes)
    districts_represented = len(votes) / total_units

    n_lost_districts = districts_represented * lose_ratio
    return n_lost_districts
