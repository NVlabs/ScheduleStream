<!--
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
# ScheduleStream algorithms

`schedulestream.algorithm` solves the problems written in the
[planning language](../language/README.md). Everything is reached through
`solve(problem, **kwargs)` in [`solver.py`](solver.py), which returns a list of solutions:

```python
from schedulestream.algorithm.solver import solve

solutions = solve(problem, lazy=False, heuristic_fn="hmax", successor_fn="offline", weight=1)
if solutions:
    solutions[0].dump()
```

The problem determines which algorithm runs. The paper's algorithms (Section IV of
[*ScheduleStream*](https://arxiv.org/abs/2511.04758)) are layered, and each one calls the next:

```
solve(problem)
 ├─ problem has no streams ── solve_finite ──────────────────────────────┐
 └─ problem has streams ───── solve_stream                               │
      ├─ "focused"  (default, the paper's LAZY-STREAM)                   │
      │    placeholders → solve_incremental ─┐                           │
      │    satisfier (bind the placeholders) │                           │
      └─ "incremental"  (the paper's EAGER-STREAM) ─┐                    │
                                                    ▼                    ▼
                                      schedule(): durative → sequential search → reschedule
                                                    ▼
                                      best_first_search (heuristics + successors)
```

| Layer | Question it answers | Files |
|---|---|---|
| [Stream algorithms](#stream-algorithms) | which values should the streams sample? | [`stream/`](stream/) |
| [Finite algorithms](#finite-algorithms) | given fixed constants, which actions and when? | [`finite/`](finite/) |
| [Scheduling](#scheduling) | how do durative actions become a timed plan? | [`schedule.py`](schedule.py), [`temporal.py`](temporal.py) |
| [Search](#search) | how is a sequential plan found? | [`search.py`](search.py), [`heuristics.py`](heuristics.py), [`successors.py`](successors.py) |

| Other file | Contains |
|---|---|
| [`instantiation.py`](instantiation.py) | Grounding streams and actions against a state: `instantiate_streams`, `instantiate_actions`, and the relaxed version that ignores deletes |
| [`utils.py`](utils.py) | `Solution`, `PartialPlan`, plan costs and applicability, and the supporter, threat and partial-order computations |

## Solving a problem

`solve` dispatches on `problem.is_stream`. Keyword arguments pass through to the algorithm, so one
call configures every layer. The ones most often set:

| Argument | Layer | Meaning |
|---|---|---|
| `algorithm` | stream or finite | `"focused"` (default) or `"incremental"` for a problem with streams; `"lazy"` (default), `"online"` or `"eager"` for one without |
| `max_time` | all | a time limit in seconds |
| `heuristic_fn` | search | `"zero"`, `"goal"`, `"hmax"`, `"hadd"`, `"hff"` or `"relaxed"` (default) |
| `successor_fn` | search | `"online"`, `"offline"`, `"relaxed"` or `"relaxed-offline"` (default) |
| `weight` | search | the heuristic weight of weighted A\*, `1` for optimal search |
| `lazy` | search | defer evaluating the heuristic until a node is expanded |
| `max_solutions`, `makespan_weight` | focused | how many solutions to collect, and how a solution is scored |
| `verbose` | all | print the search |

The examples in [`applications/minimal.py`](../applications/minimal.py) use
`lazy=False, heuristic_fn="hmax", successor_fn="offline", weight=1`, which is optimal search
over a fully grounded problem.

`solve_hierarchical(problem, functions=..., actions=...)` wraps `solve` for a problem that is
easy to plan without some conditions. It removes the functions' conditions (and the effects of the
actions), solves that high-level problem, and then refines each high-level solution in turn on the
full problem, seeded with its plan and its sampled values, until one succeeds or `max_time` runs
out. It is `problem.remove_conditions` followed by `solve`.

A solution is a [`Solution`](utils.py): a `plan`, its `cost`, whether it is `optimal`, and
`success`. A scheduled solution is a `TemporalSolution` ([`temporal.py`](temporal.py)) whose plan
has start and end times and a `makespan`, and a solution from a problem with streams is a
`StreamSolution` ([`stream/common.py`](stream/common.py)) that also records the stream outputs it
depends on (`stream_plan`).

## Stream algorithms

A problem with streams needs values for continuous parameters. Both algorithms alternate between a
search for a plan, in which the stream outputs are values the problem has so far, and calls to
streams to produce more.

### `incremental`: eager stream scheduling

[`stream/incremental.py`](stream/incremental.py), `solve_incremental`. The paper's EAGER-STREAM.
Until a plan is found:

1. search for a plan with the constants in the current initial state;
2. if there is none, instantiate every stream on every combination of the constants that satisfies
   its domain, call each instance for its next output, and add the output's certified atoms to the
   initial state.

It then returns the plan and the stream outputs that support it (`extract_stream_plan`). This is
complete, but it calls every stream that could be called whether or not the plan needs it, and a
stream that nothing constrains is called on every constant, including its own outputs, without
end. `solve_incremental` warns when stream instantiation has taken over 5 seconds.

`search_iteration` delays the first search until that many rounds of sampling have happened. The
focused algorithm uses `search_iteration=INF`, as described next.

### `focused`: lazy stream scheduling

[`stream/focused.py`](stream/focused.py), `solve_focused`. The paper's LAZY-STREAM, and the default.
It saves calls to expensive streams by planning with placeholders before it samples anything.

1. **Plan lazily.** `solve_lazy` replaces each stream's generator with one that returns a unique
   **placeholder** (`@q1`, `LazyOutput`) per input, and plans with `solve_incremental` over those.
   Procedural functions of a placeholder take their default value (`False`, `0.0`). The result is a
   schedule, written in placeholders, that would be a solution if the streams succeed.
2. **Retrace.** `retrace_solution` finds the streams whose outputs the schedule's conditions
   depend on, as a **stream plan** (a computation graph): `grasp → kin → motion`. Stream instances
   already attempted are not planned for again. The schedule together with its stream plan is a
   **skeleton**.
3. **Satisfy.** The `BatchSatisfier` binds the placeholders in each skeleton to real values by
   calling the streams, as below. A binding that satisfies all of a skeleton's constraints is a
   solution.
4. **Repeat.** If the satisfier fails to improve, the failed stream instances are excluded and the
   next iteration plans lazily again, so it produces a different skeleton. Along the way,
   the satisfier keeps sampling the skeletons it has, and each pass can produce more candidates.
5. **Reschedule.** After the loop, the best `num_reschedule` solutions are scheduled again with
   their real values (`complete_solution`, `schedule_solution`), which can shorten the makespan,
   since the collision tests are now evaluated on real trajectories.

Arguments: `max_time`, `max_skeletons`, `max_solutions` (default 1), `search_time` (per lazy
search), `satisfy_time` and `optimize_time` (the time given to the satisfier per iteration and
after the last), `num_reschedule`, `success_cost` and `makespan_weight`.

A solution may also be found in the middle of an iteration: if a skeleton's best binding is
complete but *invalid*, meaning some constraint failed on real values, the plan is scheduled again
with those values (`complete_solution`) in case a different ordering is still valid. This is how a
schedule found for parallel arms can fall back to a sequential one, as in Figure 3 of the paper.

### Satisfiers

[`stream/satisfier.py`](stream/satisfier.py) and [`stream/batch_satisfier.py`](stream/batch_satisfier.py).
A satisfier binds the values in a stream plan. The stream plan is a *skeleton*, and a **binding** is a
partial assignment of values to its placeholders. The stream plan's constraints and costs, which are
the conditions and costs of the schedule that depend on stream outputs, are checked as soon
as all of their inputs are bound, so a bad prefix is discarded early and every binding has an
accumulated cost.

| Satisfier | How it binds |
|---|---|
| `Satisfier` | a priority queue of bindings; it pops the best, calls the next stream on it for one output, and queues the extended binding |
| `BatchSatisfier` | the one `focused` uses. Per stream in the plan it takes *all* live bindings, calls the stream on every distinct input at once, joins the outputs to the bindings, scores them with the batched costs and constraints of that step, and keeps the best `max_candidates` (1000) |

The `BatchSatisfier` is what lets a GPU solver run on many hundreds of samples at once. It uses a
stream's `batch_size` (see the language's
[batched streams](../language/README.md#batched-streams)) and a function's batched definition.
It evaluates the cost of a binding at the earliest step whose outputs are all
it depends on, and sorts the bindings by it, so that the lowest-cost ones are carried to the
next stream.

### Stream plans

[`stream/stream_plan.py`](stream/stream_plan.py) works out which streams a schedule needs:
`compute_preimage` finds the atoms and costs a plan depends on, `retrace_stream_plan` and
`layer_retrace_stream_plan` trace those back through the streams that certified them, and
`reorder_stream_plan` puts the result in an order in which each stream's inputs come before it.
`visualize_stream_plan` draws it.

## Finite algorithms

[`finite/`](finite/). A problem with no streams has finitely many constants, so it only needs
scheduling. `solve_finite(problem, algorithm=...)` has three variants, which differ in when the
procedural functions are evaluated:

| `algorithm` | Behavior |
|---|---|
| `"lazy"` (default) | schedules first with each procedural function at its default value, then checks the real values of the functions the plan depends on, and keeps the best plan whose constraints hold. It loops, with the best cost as the new bound, until no better schedule is found, or `max_time` |
| `"online"` | evaluates each procedural function when search first needs it |
| `"eager"` | evaluates every static function in the problem up front, then searches |

`lazy` and `online` set a function's `lazy` flag to match. A problem solved by `focused` is
solved by `online` underneath, since its streams' outputs are already placeholders.

## Scheduling

[`schedule.py`](schedule.py) and [`temporal.py`](temporal.py). `schedule(problem)` is what every
finite algorithm calls to turn a (possibly durative) problem into a `TemporalSolution`. It
implements the paper's SCHEDULE:

1. **Compile to sequential.** `sequential_from_temporal` turns each `DurativeAction` instance
   into a start and an end action. The start action adds `Ongoing(a)` and sets the time `Remaining`
   to the duration. The end action requires `Ongoing(a)`, advances time by the remaining duration,
   and charges that time as its cost, so that a plan's cost is its **makespan**. Over conditions are
   checked, by a procedural condition, against the actions still ongoing every time one starts or
   ends.
2. **Search.** `search` finds the cheapest sequential plan for the compiled problem. The order of
   starts and ends *is* the schedule's order.
3. **Reschedule.** `reschedule_plan` runs a second search over only the actions of that plan, to
   remove delays the first search had no reason to avoid (`reschedule="search"`, the default for a
   temporal problem). `"milp"` is an alternative, and `False` skips it.
4. **Time.** `time_plan` assigns start times from the order: an action starts at the current time,
   and the time advances at an end by what remains of the duration. `recover_plan` returns the timed
   plan, a list of `TimedAction`s of `[start, end]`.

A problem with no durative actions skips steps 1 and 3 in effect, and `sequential=True` forces a
sequential plan from a temporal problem by treating durative actions as instantaneous.

Helpers in [`temporal.py`](temporal.py) convert between forms (`actions_from_timed`,
`sequential_from_timed`, `serialize_plan`), compute `get_makespan`, and build the `TemporalSolution`.

## Search

[`search.py`](search.py), the planner underneath all of the above. It runs on a sequential problem
over a state, an action's applicability and its effects.

`search(problem, algorithm="serialized")` is `best_first_search` run for a sequence of
weights:

| Function | Behavior |
|---|---|
| `best_first_search` | lazy weighted A\* with `weight`: `f = g + weight * h`. `weight=1` is A\*, and `weight=INF` is greedy |
| `iterative_best_first_search` | runs it for weights `[1, 3, 5]` in turn, stopping at the first success (`greedy=True`) |
| `anytime_best_first_search` | the same for `[5, 3, 1]`, keeping the best, so that a fast first plan is then improved |
| `serialized_search` | for a goal that is an ordered sequence, solves its parts in order, using a stage counter in the state, and otherwise is `iterative_best_first_search` |

A node is a `State`, and its edges come from a **successor function**, with a **heuristic**
estimating the cost to go.

### Successor functions

[`successors.py`](successors.py), selected by `successor_fn`:

| Name | Successors of a state |
|---|---|
| `"online"` | every action instance whose static preconditions hold, instantiated against that state |
| `"offline"` | the problem is grounded up front and each action counts its unsatisfied preconditions, so that an action becomes a successor when the count reaches zero (a Rete-style index) |
| `"relaxed"` | the actions of a relaxed plan from the state, as a macro, which makes the search greedy and fast |
| `"relaxed-offline"` | both of the above (default) |

### Heuristics

[`heuristics.py`](heuristics.py), selected by `heuristic_fn`. All but `zero` and `goal` solve a
delete relaxation of the problem, in which an atom, once true, stays true.

| Name | Estimate |
|---|---|
| `"zero"` | `0`, which makes the search uniform-cost |
| `"goal"` | the number of unsatisfied goal conditions |
| `"hmax"` | the cost to reach the most expensive goal atom (admissible, for a non-lazy search) |
| `"hadd"` | the sum of the costs of the goal atoms |
| `"hff"` | the cost of a relaxed plan, extracted by tracing back the cheapest supporters (the Fast-Forward heuristic) |
| `"relaxed"` | the default. A lazy relaxed-plan heuristic that also ignores the procedural and nested conditions, which are not known until later. The paper's modified FF |

For a temporal problem, the relaxed cost includes the times that ongoing actions have remaining.

## Choosing an algorithm

- **Default** (`solve(problem)`) is `focused` with the relaxed heuristic. Use it for a problem with
  expensive streams.
- **Optimal search** over a small, fully grounded problem: `lazy=False, heuristic_fn="hmax",
  successor_fn="offline", weight=1`.
- **`incremental`** only for a problem whose streams are cheap and whose domains are all
  constrained. It is a baseline.
- Run with an outer timeout (`timeout 300 python ...`). Grounding can run long before `max_time`
  is checked.
