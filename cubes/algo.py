import torch
torch.set_grad_enabled(False)

def vertices_to_edges_map() -> torch.Tensor:
    return torch.tensor([
        [0, 1, 4],
        [1, 2, 5],
        [2, 3, 6],
        [0, 3, 7],
        [4, 5, 9],
        [5, 9, 10],
        [6, 10, 11],
        [7, 8, 11]
    ], dtype=torch.int32)

def edges_to_vertices_map() -> torch.Tensor:
    return torch.tensor([
        [0, 3],
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [4, 7],
        [4, 5],
        [5, 6],
        [6, 7]
    ], dtype=torch.int32)

def vertices_to_faces_map() -> torch.Tensor:
    return torch.tensor([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 1, 4],
        [1, 2, 5],
        [2, 3, 5],
        [3, 4, 5],
        [1, 4, 5]
    ], dtype=torch.int32)

def faces_to_vertice_map() -> torch.Tensor:
    return torch.tensor([
        [0, 1, 2, 3],
        [0, 3, 4, 7],
        [0, 1, 4, 5],
        [1, 2, 5, 6],
        [2, 3, 6, 7],
        [4, 5, 6, 7],
    ], dtype=torch.int32)

def edges_to_faces_map() -> torch.Tensor:
    return torch.tensor([
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [1, 4],
        [1, 5],
        [2, 5],
        [3, 5],
        [4, 5],
    ], dtype=torch.int32)

def faces_to_edges_map() -> torch.Tensor:
    return torch.tensor([
        [0, 1, 2, 3],
        [0, 4, 7, 8],
        [1, 4, 5, 9],
        [2, 5, 6, 10],
        [3, 6, 7, 11],
        [8, 9, 10, 11],
    ], dtype=torch.int32)

def build_every_cube() -> torch.Tensor:
    """
    Défini par un boolean qui dit si une arrête est activée ou non.

    L'ordre des arrêtes est dans index.png
    """
    edges = torch.tensor([
        [0],
        [1],
    ], dtype=torch.bool)

    for _ in range(11):
        edges = torch.cat([edges, edges], dim=0)
        zeros = torch.zeros((edges.shape[0], 1), dtype=torch.bool)
        edges = torch.cat([zeros, edges], dim=1)
        edges[edges.shape[0]//2:, 0] = 1
    
    return edges

def visited_vertices(edges: torch.Tensor) -> torch.Tensor:
    vertices = torch.zeros((edges.shape[0], 8), dtype=torch.bool)

    for vertice_index, edges_indexes in enumerate(vertices_to_edges_map()):
        vertices[:, vertice_index] = torch.where(torch.any(edges[:, edges_indexes], dim=1), 1, vertices[:, vertice_index])
    
    return vertices

def rotate(edges: torch.Tensor, axis: str) -> torch.Tensor:
    assert axis in ("x", "y", "z")
    
    if len(edges.shape) == 1:
        batch = torch.stack([edges])
        result = rotate(batch, axis)
        return result[0]
    
    result = edges.clone()

    if axis == "x":
        result[:, 3] = edges[:, 7]
        result[:, 7] = edges[:, 11]
        result[:, 11] = edges[:, 6]
        result[:, 6] = edges[:, 3]

        result[:, 0] = edges[:, 8]
        result[:, 8] = edges[:, 10]
        result[:, 10] = edges[:, 2]
        result[:, 2] = edges[:, 0]

        result[:, 1] = edges[:, 4]
        result[:, 4] = edges[:, 9]
        result[:, 9] = edges[:, 5]
        result[:, 5] = edges[:, 1]
    elif axis == "y":
        result[:, 0] = edges[:, 1]
        result[:, 1] = edges[:, 2]
        result[:, 2] = edges[:, 3]
        result[:, 3] = edges[:, 0]

        result[:, 4] = edges[:, 5]
        result[:, 5] = edges[:, 6]
        result[:, 6] = edges[:, 7]
        result[:, 7] = edges[:, 4]

        result[:, 8] = edges[:, 9]
        result[:, 9] = edges[:, 10]
        result[:, 10] = edges[:, 11]
        result[:, 11] = edges[:, 8]
    elif axis == "z":
        result[:, 0] = edges[:, 4]
        result[:, 4] = edges[:, 8]
        result[:, 8] = edges[:, 7]
        result[:, 7] = edges[:, 0]

        result[:, 1] = edges[:, 9]
        result[:, 9] = edges[:, 11]
        result[:, 11] = edges[:, 3]
        result[:, 3] = edges[:, 1]

        result[:, 2] = edges[:, 5]
        result[:, 5] = edges[:, 10]
        result[:, 10] = edges[:, 6]
        result[:, 6] = edges[:, 2]
    
    return result

def filtrer_2d(edges: torch.Tensor) -> torch.Tensor:
    map_faces_to_edges = faces_to_edges_map()
    total_edges_per_face = torch.sum(edges[:, map_faces_to_edges], dim=2)
    total_represented_faces = torch.sum(total_edges_per_face >= 2, dim=1)
    return edges[total_represented_faces >= 2]

def filter_non_connexe(edges: torch.Tensor) -> torch.Tensor:
    vertices = visited_vertices(edges)
    map_vertices_to_edges = vertices_to_edges_map()
    map_edges_to_vertices = edges_to_vertices_map()
    
    def is_connexe(edge: torch.Tensor, vertice: torch.Tensor) -> torch.Tensor:
        visited_vertice = torch.zeros_like(vertice)
        starting_index = -1

        for i in range(vertice.shape[0]):
            if vertice[i]:
                starting_index = i
                break
        
        if starting_index == -1:
            return torch.tensor((0,), dtype=torch.bool)

        def visit(vertice_index: int | torch.Tensor):
            if visited_vertice[vertice_index]:
                return

            visited_vertice[vertice_index] = 1
            
            for neighbor_edge in map_vertices_to_edges[vertice_index]:
                if edge[neighbor_edge]:
                    for neighbor_vertice in map_edges_to_vertices[neighbor_edge]:
                        visit(neighbor_vertice)

        visit(starting_index)
        return torch.all(visited_vertice == vertice)

    filtered_edges = []

    for edge, vertice in zip(edges, vertices):
        if is_connexe(edge, vertice):
            filtered_edges.append(edge)

    return torch.stack(filtered_edges)

def filter_rotations(edges: torch.Tensor) -> torch.Tensor:
    filtered_edges = []
    unique_rotations = [
        [],
        ['z'],
        ['z', 'z'],
        ['z', 'z', 'z'],
        ['y'],
        ['y', 'z'],
        ['y', 'z', 'z'],
        ['y', 'z', 'z', 'z'],
        ['y', 'y'],
        ['y', 'y', 'z'],
        ['y', 'y', 'z', 'z'],
        ['y', 'y', 'z', 'z', 'z'],
        ['y', 'y', 'y'],
        ['y', 'y', 'y', 'z'],
        ['y', 'y', 'y', 'z', 'z'],
        ['y', 'y', 'y', 'z', 'z', 'z'],
        ['x'],
        ['x', 'z'],
        ['x', 'z', 'z'],
        ['x', 'z', 'z', 'z'],
        ['x', 'y', 'y'],
        ['x', 'y', 'y', 'z'],
        ['x', 'y', 'y', 'z', 'z'],
        ['x', 'y', 'y', 'z', 'z', 'z']
    ]

    for edge in edges:
        if not filtered_edges:
            filtered_edges.append(edge)
            continue
        
        is_good = True
        filtered_edges_tensor = torch.stack(filtered_edges) if filtered_edges else None

        for rotations in unique_rotations:
            rotated = edge

            for axis in rotations:
                rotated = rotate(rotated, axis)
            
            equal = (rotated == filtered_edges_tensor)

            if torch.any(torch.all(equal, dim=1)):
                is_good = False
                break
        
        if is_good:
            filtered_edges.append(edge)
    
    return torch.stack(filtered_edges)
