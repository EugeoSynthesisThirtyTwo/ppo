import torch
torch.set_grad_enabled(False)

def corners_to_edges_map() -> torch.Tensor:
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

def edges_to_corners_map() -> torch.Tensor:
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

def rotate(edges: torch.Tensor, axis: str) -> torch.Tensor:
    assert axis in ("x", "y", "z")
    result = edges.clone()

    if axis == "x":
        result[:, 0] = edges[:, 8]
        result[:, 8] = edges[:, 10]
        result[:, 10] = edges[:, 2]
        result[:, 2] = edges[:, 0]
    elif axis == "y":
        result[:, 4] = edges[:, 5]
        result[:, 5] = edges[:, 6]
        result[:, 6] = edges[:, 7]
        result[:, 7] = edges[:, 4]
    elif axis == "z":
        result[:, 1] = edges[:, 9]
        result[:, 9] = edges[:, 11]
        result[:, 11] = edges[:, 3]
        result[:, 3] = edges[:, 1]
    
    return result

def good_rotations() -> list[tuple[torch.Tensor, list[str]]]:
    goods = []
    edges = build_every_cube()
    rotation_hist = []

    for x in range(4):
        rotated_x = edges
        rotation_hist_x = list(rotation_hist)

        for _ in range(x):
            rotation_hist_x.append("x")
            rotated_x = rotate(rotated_x, "x")
        
        for y in range(4):
            rotated_y = rotated_x
            rotation_hist_y = list(rotation_hist_x)

            for _ in range(y):
                rotation_hist_y.append("y")
                rotated_y = rotate(rotated_y, "y")
        
            for z in range(4):
                rotated_z = rotated_y
                rotation_hist_z = list(rotation_hist_y)

                for _ in range(z):
                    rotation_hist_z.append("z")
                    rotated_z = rotate(rotated_z, "z")

                is_good = True

                for good_cube, _ in goods:
                    if torch.all(good_cube == rotated_z):
                        is_good = False
                        break
                
                if is_good:
                    goods.append((rotated_z, rotation_hist_z))
    
    return goods

good_rotations_list = good_rotations()
print(f"good rotations: {len(good_rotations_list)}")
for _, rotation_hist in good_rotations_list:
    print(rotation_hist)