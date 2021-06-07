import torch

ones = torch.ones(3)
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
print(points.storage())
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[2]
offset = second_point.storage_offset()
points_stride = points.stride()
second_point_stride = second_point.stride()
points_t = points.t()
is_equal = id(points.storage()) == id(points_t.storage())
points_t_stride = points_t.stride()