#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Standard Library
import os
import re
from typing import Any, Optional

# NVIDIA
from schedulestream.algorithm.stream.incremental import instantiate_incremental
from schedulestream.algorithm.utils import PartialPlan
from schedulestream.common.utils import current_time, elapsed_time
from schedulestream.language.connective import Conjunction
from schedulestream.language.problem import InstantiatedProblem, Problem

ANTHROPIC_MODEL = os.environ.get("SCHEDULESTREAM_ANTHROPIC_MODEL", "claude-opus-5-5")
OPENAI_MODEL = os.environ.get("SCHEDULESTREAM_OPENAI_MODEL", "gpt-5.2")


def dump_models(client) -> None:
    models = client.models.list()
    model_names = sorted(model.id for model in models.data)
    print(f"Models ({len(model_names)}): {model_names}")
    models_from_prefix = {}
    for model_name in model_names:
        if "/" in model_name:
            prefix, suffix = model_name.rsplit("/", 1)
        else:
            prefix, suffix = "", model_name
        models_from_prefix.setdefault(prefix, []).append(suffix)
    for prefix, suffixes in models_from_prefix.items():
        print(f"{prefix}) {suffixes}")


def prompt_openai(
    prompt: str, instructions: Optional[str] = None, model: str = OPENAI_MODEL, **kwargs: Any
) -> str:
    # Third Party
    from openai import OpenAI

    client = OpenAI(**kwargs)
    response = client.responses.create(model=model, instructions=instructions, input=prompt)
    return response.output_text


def prompt_anthropic(
    prompt: str, instructions: Optional[str] = None, model: str = ANTHROPIC_MODEL, **kwargs: Any
) -> str:
    # Third Party
    import anthropic

    client = anthropic.Anthropic(**kwargs)
    messages = [{"role": "user", "content": prompt}]
    response = client.messages.create(
        model=model,
        max_tokens=8096,
        system=instructions if instructions is not None else anthropic.NOT_GIVEN,
        messages=messages,
    )
    return response.content[0].text


def prompt_llm(prompt: str, **kwargs) -> str:
    return prompt_anthropic(prompt, **kwargs)


def lazily_instantiate(problem: Problem) -> InstantiatedProblem:
    problem = problem.lazy_clone(inputs=[])
    instantiate_incremental(problem)
    return problem.instantiate()


def interpret_llm(
    problem: Problem, goal_language: str, context: Optional[str] = None, **kwargs: Any
) -> Conjunction:
    instantiated = lazily_instantiate(problem)
    prompt = f"You are tasked with interpreting the following goal: {goal_language!r}."
    if context is not None:
        prompt += f"\n{context}"
    prompt += "\nRespond only with a list of the following possible conditions:"
    for language in instantiated.evaluations_from_language:
        prompt += f"\n{language}"
    print(f"\nPrompt:\n{prompt}")
    print("Waiting for LLM response...")

    start_time = current_time()
    response = prompt_llm(
        prompt,
        instructions="You are a planning agent tasked with interpreting planning goals.",
        **kwargs,
    )
    print(f"\nResponse:\n{response}\nElapsed: {elapsed_time(start_time):.3f} sec")

    matches = []
    for language, evaluations in instantiated.evaluations_from_language.items():
        language_escaped = language.replace("(", r"\(").replace(")", r"\)")
        for match in re.finditer(f"({language_escaped})", response):
            matches.append((match.start(), evaluations[0]))
    matches.sort()
    goals = [evaluation for _, evaluation in matches]
    print(f"Goals: ({len(goals)}): {goals}")
    return Conjunction(*goals)


def solve_llm(
    problem: Problem, goal_language: str, context: Optional[str] = None, **kwargs: Any
) -> Optional[PartialPlan]:
    instantiated = lazily_instantiate(problem)

    start_time = current_time()
    prompt = (
        "You are tasked with planning a sequence of actions that achieves the goal:"
        f" {goal_language!r}."
    )
    if context is not None:
        prompt += f"\n{context}"
    prompt += "\nRespond only with a list of the following actions:"
    for language in instantiated.actions_from_language:
        prompt += f"\n{language}"
    print(f"\nPrompt:\n{prompt}")
    print("Waiting for LLM response...")

    response = prompt_llm(
        prompt,
        instructions="You are a planning agent tasked with solving planning problems.",
        **kwargs,
    )
    print(f"\nResponse:\n{response}\nElapsed: {elapsed_time(start_time):.3f} sec")

    matches = []
    for language, actions in instantiated.actions_from_language.items():
        language = language.replace("(", "\(")
        language = language.replace(")", "\)")
        regex = f"({language})"
        action = actions[0]
        for match in re.finditer(regex, response):
            matches.append((match.start(), action))
    matches.sort()
    actions = [action for _, action in matches]
    partial_plan = PartialPlan(actions)
    partial_plan = partial_plan.language_plan()
    print(f"Plan ({len(partial_plan.actions)}): {partial_plan.actions}")
    return partial_plan
