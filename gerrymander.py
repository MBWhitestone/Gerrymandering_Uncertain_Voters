"""File: gerrymander.py

Authors: MBWhitestone, csirika & stefanklut
UvA Computational Social Choice Project.

This file contains an implementations of gerrymandering 
models & algorithms with uncertain voters.

May 2020
"""

import operator as op
from functools import reduce
from itertools import combinations

import numpy as np
from tqdm import tqdm

from astar import a_star


class Model():
    """Main model class."""

    def __init__(self, n_dims=0, n_dists=15, units_in_dist=100, unit_size=1,
                 shape=(15, 100)):
        """Initialize a model."""
        assert n_dims in [0, 1, 2]

        self.n_dims = n_dims
        self.n_dists = n_dists
        self.units_in_dist = units_in_dist
        self.total_units = n_dists * units_in_dist

        model = [Unit(n=unit_size) for u in range(self.total_units)]

        # For 0D use set of Units, 1D & 2D use numpy array of Units
        if n_dims == 1:
            model = np.array(model)
        elif n_dims == 2:
            assert shape[0] * shape[1] == self.total_units
            model = np.array(model).reshape(shape)
            for x, y in np.ndindex(shape):
                model[x, y].set_x_y(x, y)

        self.model = model

    def resample(self):
        """Resample voter preferences in all units of the model."""
        for unit in self.model:
            if isinstance(unit, Unit):
                unit.sample()
            else:
                for u in unit:
                    u.sample()


class Unit():
    """Neighbourhood."""

    def __init__(self, n=1, distribution='uniform'):
        """Initialize neighbourhood."""
        self.n = n
        self.distribution = distribution
        self.sample()

    def sample(self):
        """Sample voter preferences."""
        if self.distribution == 'uniform':
            self.voters = np.random.uniform(0, 1, self.n)
            self.average = np.mean(self.voters)
            self.median = np.median(self.voters)
        else:
            raise "Unknown distribution."

    def vote(self):
        """Return votes according to preferences."""
        return np.random.binomial(1, self.voters)

    def get(self, attribute):
        """Return attribute of Unit."""
        if attribute == 'average':
            return self.average
        else:
            return self.median

    def set_x_y(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        """Returns representation of Unit."""
        return f"U(n={self.n}, μ={round(self.average, 2)} " + \
               f"~={round(self.median, 2)})"

    def __lt__(self, other):
        """Needed for A* queue."""
        if isinstance(other, Unit):
            other = other.average
        return self.average < other

    def __str__(self):
        """Returns representation of Unit."""
        return self.__repr__()


class GerryMander():
    """Defines different gerrymandering attempts."""

    def __init__(self, winner=1, n_samples=100, algorithm='brute-force',
                 tie_breaking='coin'):
        """Inits the GerryMander with algorithm, tie and samples."""
        assert winner in [0, 1], "Winner should be 0 or 1"
        self.winner = winner
        self.n_samples = n_samples
        self.algorithm = algorithm
        self.tie_breaking = tie_breaking

    @staticmethod
    def partitions(n_agents, items):
        """Return partitions of items for n agents.

        Based on: https://stackoverflow.com/questions/42290859/
                  generate-all-equal-sized-partitions-of-a-set.
        """
        if n_agents == 1:
            yield [items]
        else:
            quota = len(items) // n_agents
            for indexes in combinations(range(len(items)), quota):
                remainder = items[:]
                selection = [remainder.pop(i) for i in reversed(indexes)]
                for result in GerryMander.partitions(n_agents - 1, remainder):
                    result.append(selection)
                    yield result

    def wins(self, votes):
        """Returns a winner based on votes."""
        mean = np.mean(votes)
        if mean == .5 and self.tie_breaking == "coin":
            return np.random.randint(0, 2)
        return mean > .5

    def score(self, districts):
        """Returns winner after samples elections in all districts."""
        wins = self.wins
        return np.mean([wins([wins([unit.vote() for unit in district])
                              for district in districts])
                        for sample in range(self.n_samples)])

    @staticmethod
    def nCr(n, r):
        """Returns nCr.
        See https://stackoverflow.com/questions/4941753/
            is-there-a-math-ncr-function-in-python.
        """
        r = min(r, n - r)
        numer = reduce(op.mul, range(n, n - r, -1), 1)
        denom = reduce(op.mul, range(1, r + 1), 1)
        return numer / denom

    def _brute_force(self, partitions):
        """Returns brute force best partition in model."""
        winning_score = 1 - self.winner
        winning_partition = None
        losing_partition = None
        losing_score = self.winner
        rel = op.gt if self.winner else op.lt

        for districts in tqdm(partitions):
            score = self.score(districts)

            if rel(score, winning_score):
                winning_score = score
                winning_partition = districts
            if rel(losing_score, score):
                losing_score = score
                losing_partition = districts

        return winning_score, winning_partition, losing_score, losing_partition

    def _sort(self, units, param):
        """Sort units based on param."""
        return np.array(sorted(units, key=lambda unit: unit.get(param),
                               reverse=not self.winner))

    def _packing(self, model, n_packs=1, units=None, param='median'):
        """Implements packing technique packing voters of party x together."""
        assert model.n_dists >= n_packs, "Not enough districts for packs."
        if units is None:
            units = model.model
        sorted_units = self._sort(units, param)

        pu = n_packs * model.units_in_dist
        w_score, w_districts = self._score_stack(sorted_units[:pu],
                                                 sorted_units[pu:], model)
        l_score, l_districts = self._score_stack(sorted_units[-pu:],
                                                 sorted_units[:-pu], model)

        return w_score, w_districts, l_score, l_districts

    def _score_stack(self, made_districts, rest, model):
        """Randomly assign the not already assigned."""
        made_districts = made_districts.reshape(-1, model.units_in_dist)
        todo = (np.random.permutation(rest)).reshape(-1, model.units_in_dist)

        districts = np.vstack((made_districts, todo))
        score = self.score(districts)
        return score, districts

    def _cracking(self, model, n_cracks=1, units=None, param='median'):
        """Implements cracking technique maximizing wasted votes."""
        assert model.n_dists >= n_cracks, "Not enough districts for cracks."
        if units is None:
            units = model.model
        cracks = self._sort(units, param).reshape(model.units_in_dist, -1)

        w_score, w_districts = self._score_stack(cracks[:, -n_cracks:].T,
                                                 cracks[:, :-n_cracks].T,
                                                 model)
        l_score, l_districts = self._score_stack(cracks[:, :n_cracks].T,
                                                 cracks[:, n_cracks:].T, model)

        return w_score, w_districts, l_score, l_districts

    def _pack_n_crack(self, model, packs=1, cracks=None, param='median'):
        """First pack then crack the remaining districts."""
        if cracks is None:
            cracks = model.n_dists - packs
        _, wp_districts, _, lp_districts = self._packing(model, packs,
                                                         param=param)

        w_packs = wp_districts[:packs]
        l_packs = lp_districts[:packs]

        w_units = wp_districts[packs:].flatten()
        l_units = lp_districts[packs:].flatten()
        _, wc_districts, _, _ = self._cracking(model, cracks, w_units,
                                               param=param)
        _, _, _, lc_districts = self._cracking(model, cracks, l_units,
                                               param=param)

        w_score, w_districts = self._score_stack(w_packs, wc_districts, model)
        l_score, l_districts = self._score_stack(l_packs, lc_districts, model)

        return w_score, w_districts, l_score, l_districts

    def _solve_0D(self, model, ratio=None, param=None):
        """Returns optimal 0D partitioning for winner."""
        if ratio is None:
            ratio = round(.25 * model.n_dists)
        else:
            ratio = round(ratio * model.n_dists)

        if self.algorithm == 'brute-force':
            n, s = model.total_units, 1
            while n != model.units_in_dist:
                s *= GerryMander.nCr(n, model.units_in_dist)
                n -= model.units_in_dist
            print(f'This will take me {s} iterations...')

            partitions = GerryMander.partitions(model.n_dists, model.model)
            return self._brute_force(partitions)
        elif self.algorithm == "packing":
            return self._packing(model, ratio, param)
        elif self.algorithm == "cracking":
            return self._cracking(model, param=param, n_cracks=ratio)
        elif self.algorithm == "pracking":
            return self._pack_n_crack(model, ratio, param=param)

        raise NotImplementedError

    def _solve_1D(self, model):
        """Returns optimal 1D (circle) partitioning for winner."""
        if self.algorithm == 'brute-force':
            circ = model.model.reshape(model.n_dists, model.units_in_dist)
            partitions = [np.roll(circ, i) for i in range(model.units_in_dist)]
            return self._brute_force(partitions)

        raise NotImplementedError

    def _solve_2D(self, model):
        """Returns optimal 2D partitioning for winner."""
        if self.algorithm == "a_star":
            initial = [[] for _ in range(model.n_dists)]
            nodes = list(model.model.flatten())

            w_solution = a_star((initial, nodes), model.model, self.winner,
                                model.total_units)
            w_score = self.score(w_solution[0])
            print(w_solution[0])

            l_solution = a_star((initial, nodes), model.model, 1 - self.winner,
                                model.total_units)
            l_score = self.score(l_solution[0])
            print(l_solution[0])

            return w_score, w_solution, l_score, l_solution

        raise NotImplementedError

    def solve(self, model, algorithm=None, ratio=None, param=None):
        """Returns optimal partioning in model for winner."""
        if algorithm is not None:
            self.algorithm = algorithm
        if model.n_dims == 1:
            return self._solve_1D(model)
        elif model.n_dims == 2:
            return self._solve_2D(model)

        return self._solve_0D(model, param=param, ratio=ratio)


def round_print(array):
    """Prints mean and array."""
    print(round(np.mean(array), 2), '\t', array)


def run_0D(model, gm, brute_force=True, ratio=.25, rounds=5):
    """Do multiple test with the 0D model. Don't cry when reading this."""
    assert model.n_dims == 0, "Not 0D"

    w_scores, l_scores = [], []
    wp_scores, lp_scores = [], []
    wc_scores, lc_scores = [], []
    mw_scores, ml_scores = [], []
    mwp_scores, mlp_scores = [], []
    mwc_scores, mlc_scores = [], []
    bw_scores, bl_scores = [], []

    for _ in tqdm(range(rounds)):
        if brute_force:
            bw_score, bw_part, bl_score, bl_part = gm.solve(model,
                                                            'brute-force',
                                                            ratio)
            bw_scores.append(bw_score)
            bl_scores.append(bl_score)
        else:
            gm.param = 'average'
            w_score, w_part, l_score, l_part = gm.solve(model, 'packing',
                                                        ratio)
            wc_score, wc_part, lc_score, lc_part = gm.solve(model, 'cracking',
                                                            ratio)
            wp_score, wp_part, lp_score, lp_part = gm.solve(model, 'pracking',
                                                            ratio)
            w_scores.append(w_score)
            l_scores.append(l_score)
            wp_scores.append(wp_score)
            lp_scores.append(lp_score)
            wc_scores.append(wc_score)
            lc_scores.append(lc_score)

            gm.param = 'median'
            mw_score, mw_part, ml_score, ml_part = gm.solve(model,
                                                            'packing',
                                                            ratio)
            mwc_score, mwc_part, mlc_score, mlc_part = gm.solve(model,
                                                                'cracking',
                                                                ratio)
            mwp_score, mwp_part, mlp_score, mlp_part = gm.solve(model,
                                                                'pracking',
                                                                ratio)
            mw_scores.append(w_score)
            ml_scores.append(l_score)
            mwp_scores.append(wp_score)
            mlp_scores.append(lp_score)
            mwc_scores.append(wc_score)
            mlc_scores.append(lc_score)

        model.resample()

    # Brute-force results.
    if brute_force:
        round_print(bw_scores)
        round_print(bl_scores)
        bavg = round(np.mean(np.array(bw_scores) - np.array(bl_scores)), 2)
        print(f'\nAverage brute-force difference best and worst: {bavg}')

        return bavg

    # Average results.
    for s in [w_scores, l_scores, wp_scores, lp_scores, wc_scores, lc_scores]:
        round_print(s)
    avg = round(np.mean(np.array(w_scores) - np.array(l_scores)), 2)
    avg2 = round(np.mean(np.array(wp_scores) - np.array(lp_scores)), 2)
    avg3 = round(np.mean(np.array(wc_scores) - np.array(lc_scores)), 2)
    print(f'\nAverage packing difference best and worst: {avg}')
    print(f'\nAverage pracking difference best and worst: {avg2}')
    print(f'\nAverage cracking difference best and worst: {avg3}')

    # Median results.
    for s in [mw_scores, ml_scores, mwp_scores, mlp_scores, mwc_scores,
              mlc_scores]:
        round_print(s)
    mavg = round(np.mean(np.array(mw_scores) - np.array(ml_scores)), 2)
    mavg2 = round(np.mean(np.array(mwp_scores) - np.array(mlp_scores)), 2)
    mavg3 = round(np.mean(np.array(mwc_scores) - np.array(mlc_scores)), 2)
    print(f'\nAverage median packing difference best and worst: {mavg}')
    print(f'\nAverage median pracking difference best and worst: {mavg2}')
    print(f'\nAverage median cracking difference best and worst: {mavg3}')

    return avg, avg2, avg3, mavg, mavg2, mavg3


def run_1D_2D(model, gm, rounds=5, D=1):
    """Do multiple test with the 1D or 2D model."""
    assert D in [1, 2]
    if D == 1:
        assert model.n_dims == 1, "Not 1D"
        assert gm.algorithm == 'brute-force', "1D only supports brute-force."
    elif D == 2:
        assert model.n_dims == 2, "Not 2D"
        assert gm.algorithm == "a_star", "Use \"a star\""
        assert model.total_units < 30, "better don't do this"

    w_scores, l_scores = [], []
    for _ in tqdm(range(rounds)):
        w_score, w_part, l_score, l_part = gm.solve(model)
        w_scores.append(w_score)
        l_scores.append(l_score)

        model.resample()

    # Results.
    round_print(w_scores)
    round_print(l_scores)
    avg = round(np.mean(np.array(w_scores) - np.array(l_scores)), 2)
    print(f'\nAverage {D}D difference best and worst: {avg}')

    return avg


def write_to(data, path='results.txt'):
    """Append data to path."""
    with open(path, 'a') as f:
        f.write(data + "\n")


def do_2D(g=GerryMander(algorithm="a_star")):
    """ """
    for n_dists in [3, 5, 7]:
        for units_in_dist in [3, 5, 7]:
            for unit_size in [1, 10, 100]:
                if n_dists * units_in_dist < 25:
                    m = Model(n_dims=2, unit_size=unit_size, n_dists=n_dists,
                              units_in_dist=units_in_dist,
                              shape=(n_dists, units_in_dist))
                    # a*
                    avg = run_1D_2D(m, g, D=2)
                    line1 = f"2D dists {n_dists} units_in_dist {units_in_dist}"
                    line2 = f" unit_size {unit_size} avg_score {avg}"
                    write_to(line1+line2, "results_2D.txt")


def do_1D(g=GerryMander(algorithm="brute-force"), rounds=5):
    """ """
    for n_dists in [3, 9, 27]:
        for units_in_dist in [3, 5, 9, 27]:
            for unit_size in [1, 10, 100]:
                m = Model(n_dims=1, unit_size=unit_size, n_dists=n_dists,
                          units_in_dist=units_in_dist)
                avg = run_1D_2D(m, g, D=1, rounds=rounds)
                line1 = f"1D dists {n_dists} units_in_dist {units_in_dist} "
                line2 = f"unit_size {unit_size} avg_score {avg}"
                write_to(line1+line2, "results_1D.txt")


def do_0D(g=GerryMander(algorithm="brute-force"), rounds=5):
    """ """
    for n_dists in [3, 5, 7]:
        for units_in_dist in [3, 5, 7]:
            for unit_size in [1, 10, 100]:
                line1 = f"0D dists {n_dists} units_in_dist {units_in_dist} " +\
                        f"unit_size {unit_size} "
                m = Model(n_dims=0, unit_size=unit_size, n_dists=n_dists,
                          units_in_dist=units_in_dist)

                # brute-force
                if n_dists * units_in_dist < 10:
                    line2 = f"avg_score {run_0D(m, g, True, rounds=rounds)}"
                    write_to(line1+line2, "results_0D.txt")
                # Packing, Cracking & Pracking
                for ratio in [0.25, .5, 1]:
                    line2 = f"ratio {ratio}"
                    if unit_size > 1:
                        avgs = run_0D(m, g, False, ratio=ratio, rounds=rounds)
                        line3 = f" mean pack {avgs[0]}" +\
                                f" mean crack {avgs[2]}" +\
                                f" mean prack {avgs[1]}" +\
                                f" median pack{avgs[3]}" +\
                                f" median crack {avgs[5]}" +\
                                f" median prack {avgs[4]}"
                        write_to(line1+line2+line3, "results_0D.txt")
                    else:
                        avgs = run_0D(m, g, False, ratio=ratio)
                        line3 = f" mean pack {avgs[0]} mean crack {avgs[2]}" +\
                                f" mean prack {avgs[1]}"
                        write_to(line1+line2+line3, "results_0D.txt")


if __name__ == '__main__':
    do_1D(rounds=10)
    do_0D()
    do_2D()
