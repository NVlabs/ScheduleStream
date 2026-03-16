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
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# NVIDIA
from schedulestream.common.ordered_set import OrderedSet
from schedulestream.common.queue import Queue
from schedulestream.common.utils import INF, flatten, remove_duplicates

Vertex = Any
Edge = Tuple[Vertex, Vertex]
Path = List[Vertex]
Sort = List[Vertex]


def reverse_edges(edges: Iterable[Edge]) -> List[Edge]:
    return list(map(tuple, map(reversed, edges)))


def undirected_from_edges(edges: Iterable[Edge]) -> List[Edge]:
    edges = list(edges)
    return remove_duplicates(edges + reverse_edges(edges))


def vertices_from_edges(edges: Iterable[Edge]) -> List[Vertex]:
    return remove_duplicates(flatten(edges))


def get_incoming_from_vertex(edges: Iterable[Edge]) -> defaultdict[Vertex, OrderedSet]:
    incoming_from_vertex = defaultdict(OrderedSet)
    for vertex1, vertex2 in edges:
        incoming_from_vertex[vertex2].add(vertex1)
    return incoming_from_vertex


def get_outgoing_from_vertex(edges: Iterable[Edge]) -> defaultdict[Vertex, OrderedSet]:
    outgoing_from_vertex = defaultdict(OrderedSet)
    for vertex1, vertex2 in edges:
        outgoing_from_vertex[vertex1].add(vertex2)
    return outgoing_from_vertex


def get_adjacent_from_vertex(edges: Iterable[Edge]) -> defaultdict[Vertex, OrderedSet]:
    return get_outgoing_from_vertex(undirected_from_edges(edges))


def dfs(edges: Iterable[Edge], source_vertices: Iterable[Vertex], max_depth: int = INF) -> Sort:
    outgoing_from_vertex = get_outgoing_from_vertex(edges)
    visited = set()

    def _dfs(vertex: Vertex, depth: int = 0) -> Iterable[Vertex]:
        if (depth > max_depth) or (vertex in visited):
            return
        visited.add(vertex)
        for outgoing_vertex in outgoing_from_vertex[vertex]:
            yield from _dfs(outgoing_vertex, depth=depth + 1)
        yield vertex

    order = []
    for source_vertex in source_vertices:
        order.extend(_dfs(source_vertex))
    return order


def bfs(edges: Iterable[Edge], source_vertices: Iterable[Vertex]) -> Sort:
    outgoing_from_vertex = get_outgoing_from_vertex(edges)
    visited = OrderedSet()
    queue = Queue()
    for source_vertex in source_vertices:
        queue.push(source_vertex)
    while queue:
        vertex = queue.pop()
        if vertex in visited:
            continue
        visited.add(vertex)
        for outgoing_vertex in outgoing_from_vertex[vertex]:
            queue.push(outgoing_vertex)
    return list(visited)


def search(
    edges: Iterable[Edge],
    source_vertices: Iterable[Vertex],
    use_dfs: bool = False,
) -> Sort:
    if use_dfs:
        return dfs(edges, source_vertices)
    return bfs(edges, source_vertices)


def get_ancestors(edges: Iterable[Edge], source_vertices: Iterable[Vertex]) -> Sort:
    return search(reverse_edges(edges), source_vertices)


def get_descendants(edges: Iterable[Edge], source_vertices: Iterable[Vertex]) -> Sort:
    return search(edges, source_vertices)


def get_reachable(edges: Iterable[Edge], source_vertices: Iterable[Vertex]) -> Sort:
    return search(undirected_from_edges(edges), source_vertices)


def get_components(
    edges: Iterable[Edge], vertices: Optional[Iterable[Vertex]] = None, **kwargs: Any
) -> List[List[Vertex]]:
    edges = undirected_from_edges(edges)
    if vertices is None:
        vertices = vertices_from_edges(edges)
    reached = set()
    components = []
    for vertex in vertices:
        if vertex in reached:
            continue
        component = search(edges, source_vertices=[vertex], **kwargs)
        components.append(component)
        reached.update(component)
    return components


def dfs_topological_sort(
    edges: Iterable[Edge], vertices: Optional[Iterable[Vertex]] = None
) -> Sort:
    if vertices is None:
        vertices = vertices_from_edges(edges)
    incoming_from_vertex = get_incoming_from_vertex(edges)
    source_vertices = [vertex for vertex in vertices if not incoming_from_vertex[vertex]]
    order = dfs(edges, source_vertices)
    return order[::-1]


def kahn(
    edges: Iterable[Edge], vertices: Optional[Iterable[Vertex]] = None, greedy: bool = False
) -> Iterable[Sort]:
    if vertices is None:
        vertices = vertices_from_edges(edges)
    incoming_from_vertex = get_incoming_from_vertex(edges)
    remaining = OrderedSet(vertices)
    while remaining:
        layer_vertices = []
        for vertex in remaining:
            if not any(map(remaining.__contains__, incoming_from_vertex[vertex])):
                layer_vertices.append(vertex)
                if greedy:
                    break
        if not layer_vertices:
            break
        for vertex in layer_vertices:
            remaining.remove(vertex)
        yield layer_vertices


def kahn_layers(
    edges: Iterable[Edge], vertices: Optional[Iterable[Vertex]] = None
) -> Dict[Vertex, int]:
    vertex_layers = {}
    for layer, layer_vertices in enumerate(kahn(edges, vertices, greedy=False)):
        for vertex in layer_vertices:
            vertex_layers[vertex] = layer
    return vertex_layers


def kahn_topological_sort(
    edges: Iterable[Edge],
    vertices: Optional[Iterable[Vertex]] = None,
    **kwargs: Any,
) -> Sort:
    return list(flatten(kahn(edges, vertices, **kwargs)))


def is_acyclic(edges: Iterable[Edge], vertices: Optional[Iterable[Vertex]] = None) -> bool:
    if vertices is None:
        vertices = vertices_from_edges(edges)
    sort = kahn_topological_sort(edges, vertices)
    return len(vertices) == len(sort)


def topological_sort(
    edges: Iterable[Edge],
    vertices: Optional[Iterable[Vertex]] = None,
    use_dfs: bool = False,
) -> Sort:
    if use_dfs:
        return dfs_topological_sort(edges, vertices)
    return kahn_topological_sort(edges, vertices)


def transitive_closure(edges: List[Edge]) -> List[Edge]:
    orders = []
    for vertex1 in vertices_from_edges(edges):
        for vertex2 in search(edges, source_vertices=[vertex1]):
            if vertex2 != vertex1:
                orders.append((vertex1, vertex2))
    return orders


def transitive_reduction(edges: List[Edge]) -> List[Edge]:
    assert is_acyclic(edges)
    orders = []
    outgoing_from_vertex = get_outgoing_from_vertex(edges)
    for edge in edges:
        source_vertex, target_vertex = edge
        neighbor_vertices = list(outgoing_from_vertex[source_vertex])
        if target_vertex in neighbor_vertices:
            neighbor_vertices.remove(target_vertex)
        reachable = get_descendants(edges, source_vertices=neighbor_vertices)
        if target_vertex not in reachable:
            orders.append(edge)
    return orders


def visualize_graph(
    edges: List[Edge],
    vertices: Optional[List[Vertex]] = None,
    vertex_colors: Optional[Dict[Vertex, str]] = None,
    image_path: Optional[str] = None,
    display: bool = True,
) -> str:
    # Third Party
    from pygraphviz import AGraph

    if vertices is None:
        vertices = vertices_from_edges(edges)
    vertex_colors = vertex_colors or {}
    if image_path is None:
        image_path = f"/tmp/plan.png"

    graph = AGraph(strict=True, directed=True)
    graph.node_attr["style"] = "filled"
    graph.node_attr["shape"] = "box"
    graph.node_attr["fontcolor"] = "black"
    graph.node_attr["width"] = 0
    graph.node_attr["height"] = 0.02
    graph.node_attr["margin"] = 0

    graph.graph_attr["nodesep"] = 0.1
    graph.graph_attr["ranksep"] = 0.25
    graph.graph_attr["outputMode"] = "nodesfirst"
    graph.graph_attr["dpi"] = 300

    for vertex in vertices:
        vertex_kwargs = {}
        if vertex in vertex_colors:
            vertex_kwargs["color"] = vertex_colors[vertex]
        graph.add_node(vertex, **vertex_kwargs)
    for vertex1, vertex2 in edges:
        graph.add_edge(vertex1, vertex2)

    graph.draw(image_path, prog="dot")
    print("Saved:", os.path.abspath(image_path))
    if display:
        # Third Party
        from PIL import Image

        image = Image.open(image_path)
        image.show()
    return image_path
