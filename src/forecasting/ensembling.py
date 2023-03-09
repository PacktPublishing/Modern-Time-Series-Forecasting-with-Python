import itertools
import math
import random
import warnings
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from tqdm.autonotebook import tqdm

from src.utils import ts_utils


def calculate_diversity(
    ens: List[str], diversity_matrix: pd.DataFrame, default_div: float = 1
) -> float:
    """Calculates the average diversity of an ensemble with the nxn diversity matrix

    Args:
        ens (List): The list of str with ensemble candidate names
        diversity_matrix (pd.DataFrame): Diversity matrix as a dataframe with index and columns as the candidates, diversity as the values
        default_div (float, optional): The default diversity used in cases if single element in the candidate list. Defaults to 1.

    Returns:
        float: The mean diversity of the ensemble
    """
    if len(ens) == 1:
        return default_div
    div = np.mean(
        [diversity_matrix.loc[i, j] for (i, j) in itertools.combinations(ens, 2)]
    )
    return div


def calculate_performance(
    ens: List[str],
    pred_wide: pd.DataFrame,
    target: str,
    ensemble_func: Callable = np.mean,
    metric_func: Callable = ts_utils.mae,
) -> float:
    """Calculates the performance of an ensemble

    Args:
        ens (List[str]): The list of str with ensemble candidate names
        pred_wide (pd.DataFrame): DataFrame with the forecasts and target in a wide format. Each forecast in a separate column.
        target (str): Column name of the target
        ensemble_func (Callable, optional): The function with which ensemble candidates are combined. Defaults to np.mean.
        metric_func (Callable, optional): The metric to be calculated on the resulting ensemble. metric should be of signature `metric(actuals, pred). Defaults to `MAE`.

    Returns:
        float: The performance of the ensemble
    """
    pred = ensemble_func(pred_wide[ens], axis=1)
    act = pred_wide[target]
    return metric_func(pred, act)


def generate_random_candidate(candidates: List) -> List:
    """Generates a Random candidate from alist of candidates"""
    return random.sample(candidates, 1)


def generate_best_candidate(
    objective: Callable, solution: List, candidates: List
) -> Tuple[str, float]:
    """Generates the best candidate which improves the objective

    Args:
        objective (Callable): The objective function with which to evaluate the performance of the ensemble.
            It should be a callable which takes in a list of str and returns the final objective to be minimized.
        solution (List): The existing solution/list of candidates
        candidates (List): The list of candidates which should be evaluated

    Returns:
        Tuple[str, float]: A tuple of the best new candidate and the new cost
    """
    cost = [objective(solution + [c]) for c in candidates]
    return [candidates[np.argmin(cost)]], np.min(cost)


def _initialize(candidates: List, objective: Callable, init: str):
    """Initializes the initial list of candidates either by picking the best or randomly"""
    if init == "best":
        cost = [objective([c]) for c in candidates]
        return [candidates[np.argmin(cost)]], np.min(cost)
    elif init == "random":
        c = generate_random_candidate(candidates)
        return c, objective(c)
    else:
        raise ValueError("`init` can either be `random` or `best`")


# Greedy Optimization local search algorithm
def greedy_optimization(
    objective: Callable, candidates: List[str], verbose: bool = True
) -> Tuple[List[str], float]:
    """Performs Greedy Optimization to find the best ensemble

    Args:
        objective (Callable): The objective function with which to evaluate the performance of the ensemble.
            It should be a callable which takes in a list of str and returns the final objective to be minimized.
        candidates (List[str]): Candidates for ensembling as a list of str
        verbose (bool, optional): Whether to print progress or not. Defaults to True.

    Returns:
        Tuple[List[str], float]: A tuple of the best solution and the new cost
    """
    candidates = candidates.copy()
    # generate and evaluate an initial point as the best performing algorithm
    solution, solution_eval = _initialize(candidates, objective, init="best")
    # Removing the candidate from remaining candidates
    candidates.remove(solution[0])
    # run the greedy_optimiaztion
    while len(candidates) > 0:
        # generate and evaluate new best point
        _candidate, candidate_eval = generate_best_candidate(
            objective, solution, candidates
        )
        # Add the candidate to the solution to get the new candidate
        candidate = solution + _candidate
        # check if we should keep the new point
        if candidate_eval <= solution_eval:
            # store the new point
            solution, solution_eval = candidate, candidate_eval
            # Removing the candidate from remaining candidates
            candidates.remove(_candidate[0])
            if verbose:
                # report progress
                print(f"Solution: {solution} | Best Score: {solution_eval}")
        else:
            if verbose:
                print("Solution cannot be improved further. Stopping optimization.")
            break
    return (solution, solution_eval)


# hill climbing local search algorithm
def stochastic_hillclimbing(
    objective: Callable,
    candidates: List[str],
    n_iterations: int = None,
    init: str = "best",
    verbose: bool = True,
    random_state: int = 42,
) -> Tuple[List[str], float]:
    """Performs stochastic hill-climb to find the best ensemble out of a list of candidates

    Args:
        objective (Callable): The objective function with which to evaluate the performance of the ensemble.
            It should be a callable which takes in a list of str and returns the final objective to be minimized.
        candidates (List[str]): Candidates for ensembling as a list of str
        n_iterations (int): Number of iterations to run the hill-climb for. If not given will revert to a heuristic: len(candidates)*2
        init (str): Specifies how to generate initial solution. Options are `best` and `random`.
        verbose (bool, optional): Whether to print progress or not. Defaults to True.
        random_state (int): To maintain reproduceability. Defaults to 42

    Returns:
        Tuple[List[str], float]: A tuple of the best solution and the new cost
    """
    random.seed(random_state)
    n_iterations = len(candidates) * 2 if n_iterations is None else n_iterations
    if n_iterations < len(candidates):
        warnings.warn(
            "`n_iterations` lower than number of candidates. Consider increasing the `n_iterations`."
        )
    # Making a copy of the list to make sure we do not alter the original
    candidates = candidates.copy()
    # generate and evaluate an initial point
    solution, solution_eval = _initialize(candidates, objective, init)
    # Removing the candidate from remaining candidates
    candidates.remove(solution[0])
    # run the hill climb
    for i in range(n_iterations):
        # take a step
        _candidate = generate_random_candidate(candidates)
        candidate = solution + _candidate
        # evaluate candidate point
        candidate_eval = objective(candidate)
        # check if we should keep the new point
        if candidate_eval <= solution_eval:
            # store the new point
            solution, solution_eval = candidate, candidate_eval
            # Removing the candidate from remaining candidates
            candidates.remove(_candidate[0])
            if verbose:
                # report progress
                print(
                    f"Iteration: {i}: Solution: {solution} | Best Score: {solution_eval}"
                )
        else:
            if verbose:
                print(
                    f"Iteration: {i}: Iteration did not improve the score. Solution: {solution} | Best Score: {solution_eval}"
                )
    return (solution, solution_eval)


def _decay_temperature(
    current_temp: float, alpha: float, kind: str = "linear"
) -> float:
    """Performs the temperature decay in simulated annealing"""
    if kind == "linear":
        new_temp = current_temp - alpha
    elif kind == "geometric":
        new_temp = current_temp / alpha
    return new_temp


# D.S. Johnson, C.R. Aragon, L.A. McGeoch, and C. Schevon, “Optimization by simulated annealing: Anexperimental evaluation; part I, graph partitioning,” Operations Research
def initialize_temperature_range(
    objective: Callable,
    candidate_pool: list[str],
    p_range: Tuple[float, float],
    n_iterations: int = 100,
) -> Tuple[float, float]:
    """Initializes Temperature range by estimating the initial temperature using the method proposed by D.S. Johnson et al. in

    Args:
        objective (Callable): The objective function with which to evaluate the performance of the ensemble.
            It should be a callable which takes in a list of str and returns the final objective to be minimized.
        candidate_pool (list[str]): Candidates for ensembling as a list of str
        p_range (Tuple[float, float]): Probability range as a tuple (start, end). This is the probability with which a worse solution is accepted in the simulated annealing.
        n_iterations (int, optional): Number of samples to run to estimate average error delta. Defaults to 100.

    Returns:
        Tuple[float,float]: Returns the temerature range (start_temperature, end_temperature)
    """
    diff_l = []
    _candidate_pool = candidate_pool.copy()
    candidates = generate_random_candidate(_candidate_pool)
    candidate_score = objective(candidates)
    for _ in tqdm(range(n_iterations)):
        if len(_candidate_pool) == 0:
            _candidate_pool = candidate_pool.copy()
            candidates = generate_random_candidate(_candidate_pool)
            candidate_score = objective(candidates)
        cand = generate_random_candidate(_candidate_pool)
        candidates += cand
        _candidate_pool.remove(cand[0])
        diff = candidate_score - objective(candidates)
        diff_l.append(diff)
    avg_diff = np.median(np.abs(diff_l))
    t_range = (-avg_diff / math.log(p_range[0]), -avg_diff / math.log(p_range[1]))
    return t_range


def _calculate_decay(
    t_range: Tuple[float, float], n_iterations: int, temperature_decay: str
):
    """Calculates the decay with which the temperature can be reduced each iteration, either linearly or in a geometric fashion"""
    if temperature_decay == "linear":
        alpha = (t_range[0] - t_range[1]) / (n_iterations - 1)
    elif temperature_decay == "geometric":
        alpha = math.pow((t_range[0] / t_range[1]), 1 / (n_iterations - 1))
    return alpha


# hill climbing local search algorithm
def simulated_annealing(
    objective: Callable,
    candidates: List[str],
    n_iterations: int,
    p_range: Tuple[float, float] = (0.7, 0.001),
    t_range: Tuple[float, float] = None,
    init: str = "best",
    temperature_decay: str = "linear",
    verbose: bool = True,
    random_state: int = 42,
) -> Tuple[List[str], float]:
    """Performs simulated annealing to find the best ensemble out of a list of candidates

    Args:
        objective (Callable): The objective function with which to evaluate the performance of the ensemble.
            It should be a callable which takes in a list of str and returns the final objective to be minimized.
        candidates (List[str]): Candidates for ensembling as a list of str
        n_iterations (int): Number of iterations to run the hill-climb for. If not given will revert to a heuristic: len(candidates)*2
        p_range (Tuple[float, float]): Probability range as a tuple (start, end). This is the probability with which a worse solution
            is accepted in the simulated annealing. Temperature range is inferred from p_range during optimization
        t_range (Tuple[float, float]): Temperature range as a tuple (start, end). This is the raw temperatures used while annealing, if we want to set it explicitly.
            If this is provided, `p_range` is ignored
        init (str): Specifies how to generate initial solution. Options are `best` and `random`.
        temperature_decay (str): Specifies how to decay the temperature. `current_temp-alpha` for `linear` and `current_temperature/alpha` when `geometric`.
        verbose (bool, optional): Whether to print progress or not. Defaults to True.
        random_state (int): To maintain reproduceability. Defaults to 42

    Returns:
        Tuple[List[str], float]: A tuple of the best solution and the new cost
    """
    random.seed(random_state)
    if p_range is None and t_range is None:
        raise ValueError("Either t_range or p_range should be given as an input")
    n_iterations = min(n_iterations, int(len(candidates) * 1.2))
    if t_range is None:
        if verbose:
            print("Finding optimum temperature range")
        t_range = initialize_temperature_range(objective, candidates, p_range)

    # Reduction in each iteration
    alpha = _calculate_decay(t_range, n_iterations, temperature_decay)
    # Making a copy of the list to make sure we do not alter the original
    candidates = candidates.copy()
    # generate and evaluate an initial point
    best_solution, best_solution_eval = _initialize(candidates, objective, init)
    # Removing the candidate from remaining candidates
    candidates.remove(best_solution[0])
    # set that as the current working  solution
    # current_solution, current_solution_eval = best_solution, best_solution_eval
    # setting temperature to be starting temp
    current_temp = t_range[0]
    # run the hill climb
    for i in range(n_iterations):
        # take a step
        _candidate = generate_random_candidate(candidates)
        candidate = best_solution + _candidate
        # evaluate candidate point
        candidate_eval = objective(candidate)
        # difference between candidate and current point evaluation
        diff = best_solution_eval - candidate_eval
        # If the new solution is better accept it
        # If the new solution not better, accept it with a probability of e^(-cost/temp)
        if diff > 0 or random.uniform(0, 1) < math.exp(-abs(diff) / current_temp):
            # Accepting the new solution
            (best_solution, best_solution_eval) = (candidate, candidate_eval)
            candidates.remove(_candidate[0])
            if verbose:
                # report progress
                print(
                    f"Iteration: {i}: Solution: {best_solution} | Best Score: {best_solution_eval}"
                )
        else:
            if verbose:
                print(
                    f"Iteration: {i}: Iteration did not improve the score. Solution: {best_solution} | Best Score: {best_solution_eval}"
                )
        current_temp = _decay_temperature(current_temp, alpha, temperature_decay)
        if len(candidates) == 0:
            print("Ran out of candidates. Stopping the optimization")
            break
    return (best_solution, best_solution_eval)


def find_optimal_combination(
    candidates: List[str],
    pred_wide: pd.DataFrame,
    target: str,
    metric_fn: Callable = ts_utils.mae,
) -> List[float]:
    """Runs an optimization to find the best weights with which the candidate forecasts can be combined in an average

    Args:
        candidates (List[str]): The list of str with ensemble candidate names
        pred_wide (pd.DataFrame): DataFrame with the forecasts and target in a wide format. Each forecast in a separate column.
        target (str): Column name of the target
        metric_fn (Callable, optional): The metric to be calculated on the resulting ensemble. metric should be of signature `metric(actuals, pred). Defaults to `MAE`.

    Returns:
        List[float]: The optimal weights
    """

    def loss_function(weights):
        # fc = np.average(pred_wide[candidates], weights=weights, axis=1)
        # This is faster
        fc = np.sum(pred_wide[candidates].values * np.array(weights), axis=1)
        return metric_fn(pred_wide[target].values, fc)

    opt_weights = optimize.minimize(
        loss_function,
        x0=[1 / len(candidates)] * len(candidates),
        constraints=({"type": "eq", "fun": lambda w: 1 - sum(w)}),
        method="SLSQP",  # 'SLSQP', Nelder-Mead
        bounds=[(0.0, 1.0)] * len(candidates),
        options={"ftol": 1e-10},
    )["x"]
    return opt_weights
