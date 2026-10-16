<!--
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
# The ScheduleStream planning language

`schedulestream.language` is the Python language for writing planning & scheduling problems with
**samplers**. It is the layer underneath every application.

A problem is declarative where it can be, as functions, predicates, actions and their conditions,
and procedural where it must be, as Python functions that test, score or sample continuous values
(configurations, poses, trajectories).

The language is introduced in [*ScheduleStream: Temporal Planning with Samplers for GPU-Accelerated
Multi-Arm Task and Motion Planning & Scheduling*](https://arxiv.org/abs/2511.04758) (Section III).

| Concept | Class | File |
|---|---|---|
| Relations and state variables | `Predicate`, `Function`, `NumericFunction` | [`predicate.py`](predicate.py), [`function.py`](function.py) |
| Initial state | `State`, or a list of assignments | [`state.py`](state.py) |
| Goal conditions | a list or `Formula` of conditions | [`problem.py`](problem.py), [`connective.py`](connective.py) |
| Instantaneous action | `Action` | [`action.py`](action.py) |
| Durative action | `DurativeAction` | [`durative.py`](durative.py) |
| Procedural function | `Function(definition=...)`, `NumericFunction` | [`function.py`](function.py) |
| Sampler | `Stream`, `P.stream(...)` | [`stream.py`](stream.py), [`generator.py`](generator.py) |
| Problem | `Problem` | [`problem.py`](problem.py) |

The reference is [`applications/minimal.py`](../applications/minimal.py), the bimanual example of
the paper. Each concept below builds on the previous one, ending with the problem that assembles
them. It runs with:

```bash
PYTHONPATH=src python -m schedulestream.applications.minimal
```

## Constants

A problem is over a universe of **constants**, atomic values of any Python type:

- discrete constants, such as the strings `"arm1"` and `"obj1"`;
- continuous constants, such as lists, NumPy arrays or (PyTorch) tensors for configurations, poses
  and trajectories.

Constants are wrapped by identity: two equal lists are two different constants, so use tuples where
equal values should be the same constant.

## Functions and predicates

Following Functional STRIPS, state is described by **functions**: a relation from a tuple of input
constants to one output constant. A **predicate** is a function whose output is a Boolean.

```python
Object = Predicate("?obj")                                   # a type
Placement = Predicate("?obj ?p")                             # a static relation
Pose = Function("?obj", condition=[Object("?obj")])          # a state variable: ?obj -> pose
Holding = Function("?arm", condition=[Arm("?arm")])
```

- **Parameters** are strings beginning with `?`, space separated: `"?arm ?q"`.
- **`condition=`** is the *domain condition*: atoms the arguments are guaranteed to satisfy.
  `Pose("?obj")` is only defined for an `Object`. `OUTPUT` stands for the function's value, as in
  `At("?arm")`'s condition `Conf("?arm", OUTPUT)`, which says the arm's configuration is a valid
  `Conf` for that arm.
- **Closed world:** an unassigned function evaluates to `None`, and an unasserted predicate to
  `False`.
- **Static vs fluent:** a function that no action assigns is static, and its values are facts. One
  that an action assigns is fluent, and its values form the state.
- **Naming:** call `rename_anonymous(locals())` after declaring functions and actions to name each
  after its Python variable. Otherwise it is `Function3`, and so on.

### Terms and atoms

Calling a function on constants or parameters makes a **term**, and comparing or assigning it makes
a condition or an effect:

| Expression | Means | As |
|---|---|---|
| `Holding(arm1) == obj1` | the term currently equals `obj1` | a condition, or a goal |
| `Holding("?arm") == None` | the term is unassigned | a condition |
| `Holding("?arm") <= "?obj"` | the term is assigned `?obj` | an effect, or an initial assignment |
| `Placement(obj1, p1)` | the atom holds | a condition, or a static fact |
| `~Placement("?obj ?p")` | the atom does not hold | a condition |

A predicate may be called with one string, `Kin("?arm ?q ?obj ?g ?p")`, or with separate
arguments. A function with a value must be called with separate arguments, as in
`Duration("?t")` and `Pose(Holding("?arm"))`. Terms can be nested.

## Initial state

A **state** is the set of values of the functions, and the initial state `I` is where planning
starts. It is written as a list of the assignments and facts from the table above:

```python
initial = [
    At(arm1) <= q1, At(arm2) <= q2,                  # fluent assignments
    Holding(arm1) <= None, Holding(arm2) <= None,
    Pose(obj1) <= p1, Pose(obj2) <= p2,
    Placement(obj1, p1), Placement(obj2, p2),        # static facts
    Attached(obj1) <= None, Attached(obj2) <= None,
]
```

- **Fluents** are assigned with `<=`. These values change as actions apply their effects.
- **Facts** are atoms of static predicates, such as `Placement(obj1, p1)`. They are never changed
  by an action, but streams add new ones as they certify the values they sample.
- **Closed world:** anything not listed is `None` or `False`, so `Holding(arm1) <= None` states
  the default explicitly and changes nothing. It is still worth writing, for the reader.
- **Other forms:** `Problem(initial=...)` also accepts a `State`, or a dict from terms to values,
  `{At(arm1): q1, ...}`.

## Goal conditions

The goal says what a plan must achieve. It is written with the same conditions as an action's
precondition and is checked against the final state:

```python
goal = [Holding(arm1) == obj1, Holding(arm2) == obj2]        # a conjunction
goal = [Holding(arm1) == obj1, Holding(arm2) == None]        # a function may be required unassigned
```

- **A list is a conjunction:** every condition must hold.
- **Formulas** combine conditions with `&` and `|`. A disjunction, such as
  `(Holding(arm1) == obj1) | (Holding(arm2) == obj1)`, compiles to one zero-cost goal action per
  disjunct, so planning only has to reach a single goal atom. This is how a task can leave the
  assignment of arms to objects to the planner.
- **Goals are over constants.** A goal on a sampled value, such as `Pose(obj1) == p3`, can only be
  reached by a stream that outputs that exact constant. Make the target a fact, as in
  `Placement(obj1, p3)` (see [Rules of thumb for streams](#rules-of-thumb-for-streams)).

## Actions

An `Action` is instantaneous. It applies to a state in which its `precondition` holds, and updates
the state with its `effect`.

| Field | Holds |
|---|---|
| `parameters` | the action's variables, such as `"?arm ?q ?obj"`, all bound when it is applied |
| `precondition` | conditions that must all hold in the current state: function equalities, predicate atoms, negations, tests |
| `effect` | assignments `f(...) <= value` that give functions their new values |
| `cost` | what applying it adds to the plan's cost: a number or numeric term, by default a tiny positive constant |

- **Preconditions** read the state. `Holding("?arm") == None` requires that the arm holds nothing,
  and `Pose("?obj") == "?p"` ties `?p` to the object's current pose. A condition over only static
  functions never depends on the state, so it only restricts which parameter values are valid.
- **Effects** are applied all at once, with every value read from the state *before* the action.
  Assigning `None` unassigns a function, so `Holding("?arm") <= None` is how a place releases an
  object. A function that no effect mentions keeps its value, so effects only say what changes. A
  function is fluent if some action assigns it.
- **Cost** may be a [procedural function](#procedural-functions) of the parameters. `pick` below is
  free, since only motion has a cost that matters.

Together they say what an action needs and what it changes. In `pick` below, the arm must be at
`?q`, the object at `?p`, and both unoccupied. Afterward the object is at the grasp `?g`, attached
to the arm, and the arm holds the object, while the arm's configuration is left as it was.

```python
pick = Action(
    parameters="?arm ?q ?obj ?g ?p",
    precondition=[
        Kin("?arm ?q ?obj ?g ?p"),      # a kinematic solution: FK(?q) * ?g = ?p
        At("?arm") == "?q",
        Pose("?obj") == "?p",
        Attached("?obj") == None,
        Holding("?arm") == None,
    ],
    effect=[Pose("?obj") <= "?g", Attached("?obj") <= "?arm", Holding("?arm") <= "?obj"],
    cost=0.0,
)
```

All of the action's parameters are bound at once, and a condition over only static functions,
like `Kin(...)`, constrains which values the parameters may take. Where the parameters' values come
from is the job of streams.

Cost and domain conditions don't have to be written out: a parameter's types are implied by the
domain conditions of the functions it is passed to.

## Durative actions

Plans in ScheduleStream are **schedules**: actions overlap in time rather than following one
another. A `DurativeAction` starts at one time and ends later, and each of its two events changes
the state. It splits an `Action`'s precondition and effect by *when* they apply:

| Field | Holds |
|---|---|
| `start_condition` | in the state just before it starts |
| `over_condition` | the whole time it is ongoing, exclusive of its start and end |
| `end_condition` | in the state just before it ends |
| `start_effect` | the state change when it starts |
| `end_effect` | the state change when it ends |
| `cost` | what the action adds to the plan's cost, counted once |
| `min_duration` | the least time between its start and end, a number or a term |

- **Preconditions** of an `Action` become three: the `start_condition`, which holds before it
  starts, the `end_condition`, which holds before it ends, and the `over_condition`, which holds
  in between.
- **Effects** become two: the `start_effect` happens when it starts and the `end_effect` when it
  ends.
- **Cost** is the same as an `Action`'s. The `min_duration` is new, and sets the time between the
  start and the end.

An `Action` is the special case with only a start: no time passes, and its precondition and
effect are the start condition and effect. Conversely, a durative action behaves as the two
instantaneous actions `a.start` and `a.end`:

- **Starting** requires the `start_condition` and `over_condition`, and that it is not already
  ongoing, and applies the `start_effect`. The action is then ongoing, with `min_duration` remaining.
- **While ongoing**, its `over_condition` has to hold in the state at every event that happens
  before it ends, including other actions' starts and ends.
- **Ending** requires the `end_condition`, and applies the `end_effect`. Time elapses by what
  remains of the duration, and the schedule gets its end time from it.

```python
move = DurativeAction(
    parameters="?arm ?q1 ?t ?q2",
    start_condition=[Motion("?arm ?q1 ?t ?q2"), At("?arm") == "?q1"],
    start_effect=[At("?arm") <= "?t"],                                  # somewhere along ?t
    over_condition=[~ObjCollision("?t", Pose(o)) for o in [obj1, obj2]]
                 + [~ArmCollision("?t", At(a)) for a in [arm1, arm2]],
    end_effect=[At("?arm") <= "?q2"],
    min_duration=Duration("?t"),
    cost=0.0,
)
```

Here starting `move` sets `At(arm)` to the *trajectory*, so "the arm is somewhere along `?t`".
While it runs, `over_condition` has to keep holding for every other action that is concurrent with
it, which is how it checks collisions against another arm's trajectory. Ending it sets `At(arm)`
to the end configuration. Its duration is a [procedural function](#procedural-functions) of the
trajectory.

What the over condition sees is what the state looks like at *each other event*: when another arm
is itself mid-move, `At(arm2)` is its trajectory, and `ArmCollision` compares two trajectories.
When it is stationary, `At(arm2)` is a configuration, and the test compares a trajectory with a
configuration. This is why one `ArmCollision` test covers both the parallel and the sequential
schedules (Figure 3 of the paper).

A solution minimizes the **makespan**, the time at which the last action ends. A problem with no
`DurativeAction`s is ordinary sequential planning and minimizes plan cost.

## Procedural functions

`definition=` replaces a declared relation with Python code evaluated on demand, so the function
needs no enumerated values. It is the way to express values that are defined by a computation
over continuous inputs:

```python
Duration = NumericFunction("?t", definition=lambda t: len(t) * 0.1)   # -> float, default 0.0
ObjCollision = Predicate("?t ?p", definition=lambda t, p: collides(t, p))  # -> bool
```

- A `NumericFunction` has default value `0.0`, and is used for durations and costs.
- A procedural predicate is a test. `~ObjCollision("?t", Pose(o))` is a condition that is checked
  once the trajectory `?t` is known.
- Procedures are evaluated on the current state when a term nests a state function, as in
  `Pose(o)` above, so the test sees the object's pose when the action runs.
- **Batching:** `batch_size=N` makes the definition take a list of input tuples, one per call, and
  return a list with one result per input. A GPU collision checker is then called once per batch
  rather than once per input.

## Streams

Streams generate the continuous parameters that actions need. They are
[PDDLStream](https://arxiv.org/abs/1802.08705) streams: a *conditional sampler* that takes input
constants and yields output constants certified to satisfy a predicate.

A stream is derived from a predicate by choosing which of its parameters are inputs. The rest are
outputs:

```python
Grasp = Predicate("?arm ?obj ?g", condition=[Arm("?arm"), Object("?obj")])
grasps = Grasp.stream(inputs="?arm ?obj", conditional_generator=from_unary_gen_fn(sample_grasps))
```

This reads: given an `Arm` and an `Object`, `grasps` yields values `?g` such that
`Grasp(arm, obj, g)` is true. The predicate's condition does the rest of the specification, with
the part about the inputs, `Arm("?arm") & Object("?obj")`, becoming the stream's **domain**,
and the rest its **certified** atoms. The `Kin` stream above is called with a grasp *produced by*
`grasps`, which is how streams compose into a pipeline: `grasps → kin → motion`.

The `conditional_generator` takes the input values and returns an iterator that yields a list of
output tuples for each step, possibly empty, until the stream is exhausted. [`generator.py`](generator.py)
has adapters so that a plain Python function is enough:

| Adapter | From a Python function that... |
|---|---|
| `from_unary_fn` | returns one output value, or `None` |
| `from_unary_gen_fn` | is a generator of single output values (`yield q`) |
| `from_fn`, `from_gen_fn` | returns, or yields, an output tuple for several outputs |
| `from_test` | returns a Boolean, for a stream with no outputs |
| `batch_fn_from_list_gen_fn` | is a stream above, lifted to a batch (below) |

Two options on a stream deserve mention:

- `priority=` orders when a stream is called relative to the others. A negative priority, such as
  `priority=-INF` for an expensive motion planner, defers it.
- `conditional_generator=None` makes a placeholder stream that only ever returns lazy values,
  which is what `minimal.py` does to check that a problem is solvable before any sampler exists.

### Batched streams

For a GPU solver, `batch_size=N` on `P.stream(...)` makes the generator take a *list* of up to `N`
input tuples, and yield `(input, output)` pairs, in any order and with any number of outputs per
input. An input with no pair has no output. `batch_fn_from_list_gen_fn` lifts an unbatched sampler
into this form for testing.

```python
Kin.stream(inputs="?robot ?obj ?grasp ?placement",
           conditional_generator=ik_fn,           # ik_fn(inputs_list) -> [(inputs, (q,)), ...]
           batch_size=128)
```

### Rules of thumb for streams

- **Constrain every input.** A stream whose input is not constrained by a domain condition is
  tried on *every* constant in the problem, including its own outputs, and planning may never
  finish. The solver warns after 5 seconds of stream instantiation. Give the predicate a
  `condition` such as `[Arm("?arm"), Object("?obj")]` as `Grasp` does.
- **Sampled values equal no constant.** A goal on a specific value, such as `Pose(obj1) == p3`,
  can only be reached by a stream that outputs that exact constant. Make the target a fact, as in
  `Placement(obj1, p3)`.
- **Prefer several low-arity streams, tests and functions to one high-arity one.** Their
  instances are shared between actions and cached, and a failure names the culprit.

## Problems and solving

Definition 1 in the paper: a problem is `⟨I, G, A, S⟩`, an initial state, goal conditions, a set of
actions and a set of streams. `solve` dispatches to a finite search for a problem with no streams
and to a stream algorithm otherwise.

`Problem` also offers inspection and variants of the problem for debugging:
`problem.dump()`, `problem.lazy_clone()` (every stream replaced by one that returns placeholders,
to check that the model alone is solvable) and `problem.remove_conditions([...])` (drop a
function's conditions, as in `solve_hierarchical`).

### Putting it together

Every concept above, including the goal, appears in one problem. Two arms must each hold an
object. The arms can move at the same time, as long as their motions don't collide.

```python
Arm = Predicate("?arm")
Conf = Predicate("?arm ?q")
At = Function("?arm", condition=[Arm("?arm"), Conf("?arm", OUTPUT)])    # a state variable

Kin = Predicate("?arm ?q ?obj ?g ?p",
                condition=[Conf("?arm ?q"), Placement("?obj ?p"), Grasp("?arm ?obj ?g")])
Motion = Predicate("?arm ?q1 ?t ?q2", condition=[Conf("?arm ?q1"), Conf("?arm ?q2")])
Duration = NumericFunction("?t", definition=lambda t: 1e-2)             # a procedure

move = DurativeAction(
    parameters="?arm ?q1 ?t ?q2",
    start_condition=[Motion("?arm ?q1 ?t ?q2"), At("?arm") == "?q1"],
    start_effect=[At("?arm") <= "?t"],                                  # somewhere along ?t
    over_condition=[~ObjCollision("?t", Pose(o)) for o in [obj1, obj2]]
                 + [~ArmCollision("?t", At(a)) for a in [arm1, arm2]],
    end_effect=[At("?arm") <= "?q2"],
    min_duration=Duration("?t"),
    cost=0.0,
)

streams = [Grasp.stream(inputs="?arm ?obj", conditional_generator=...),
           Kin.stream(inputs="?arm ?obj ?g ?p", conditional_generator=...),
           Motion.stream(inputs="?arm ?q1 ?q2", conditional_generator=...)]

problem = Problem(initial=initial, goal=[Holding(arm1) == obj1, Holding(arm2) == obj2],
                  actions=[move, pick], streams=streams)
solutions = solve(problem, lazy=False, heuristic_fn="hmax", successor_fn="offline")
solutions[0].dump()
```

The solution is a schedule, here with both arms moving in parallel:

```
Plan (4): [move(arm1, [0.0, 1.0], @t872, @q936)[0.00, 0.01], move(arm2, [1.0, 1.0], @t752, @q0)[0.00, 0.01],
           pick(arm1, @q936, obj1, @g192, [0.0, 0.0])[0.01, 0.01], pick(arm2, ...)[0.01, 0.01]]
```

Each action is annotated with its `[start, end]` time. Values the planner sampled, such as the
trajectory `@t872`, are printed with an `@` prefix.

### How the solver works

This is an overview. The [algorithm README](../algorithm/README.md) describes each algorithm, its
search and its arguments in detail.

Each solution is found by alternating a scheduling phase and a sampling phase:

1. **Schedule.** A durative action `a` compiles to an instantaneous `a.start` and `a.end` that
   tracks which actions are `Ongoing` and their time `Remaining`. Sequential search over these
   events, with an FF-style heuristic, finds the order of every start and end, and the schedule
   follows from it. This is how a durative action's `over_condition` is checked against concurrent
   ones.
2. **Sample.** The streams are called so that the schedule's parameters have real values.

The *lazy* algorithms (the default, `algorithm="focused"` in the code) plan with **placeholder** values
first, written `@q1`, where each stream's output is a unique stand-in and each procedural function
takes its default value (`False`, `0.0`). Only for a schedule that is found does the planner trace
back which stream calls produce its placeholders, and call them, substituting the real values. When
a call fails, the planner goes back to try another schedule. The *eager* algorithm instead calls
every stream before every search. Since streams are the expensive part of robotics problems, the
lazy ones are preferred.

Hence the output of a failed problem is not always the model's fault. It may be that the
placeholders were consistent but no real values were: a stream that never succeeds, or a test that
fails for every candidate.

## Other files

| File | Contains |
|---|---|
| [`expression.py`](expression.py), [`connective.py`](connective.py) | The `Formula` operators `&`, `|`, `~`, and `Conjunction`, `Disjunction`, `Sum` |
| [`argument.py`](argument.py) | `Parameter`, `Constant`, `OUTPUT`, and wrapping values as constants |
| [`effect.py`](effect.py), [`constraint.py`](constraint.py) | Assignments, and the costs and constraints that stream satisfiers optimize |
| [`lazy.py`](lazy.py) | `LazyOutput`, the placeholders, and lazy versions of functions |
| [`anonymous.py`](anonymous.py) | `rename_anonymous` |

## Common mistakes

- Using `is None` or `not` on a term: `Holding("?arm") == None` builds a condition, but `is None`
  tests the Python object.
- Calling a function with a value as one string, `Pose("?obj ?p")`: only a predicate splits
  one string into its arguments.
- Forgetting `rename_anonymous(locals())`, which leaves anonymous names in logs.
- A stream input that nothing constrains (see above).
- A goal on a sampled value that is not a fact.
- Mutating a constant, such as a list or tensor, after it is in the state. Constants are compared
  by identity.
