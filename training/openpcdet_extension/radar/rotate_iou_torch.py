"""Rotated-box IoU without numba.cuda, used as the CPU/no-GPU fallback in radar_eval.

numba 0.61's ctypes CUDA driver binding segfaults during context creation against
the NVIDIA 580.x / CUDA 13 driver on this cluster, so importing ``rotate_iou``
crashes the interpreter before any evaluation runs. The rotated-box intersection
it computes is already available from the compiled ``iou3d_nms`` PyTorch
extension that training uses for NMS, so this module routes the evaluator through
that op instead.

Convention mapping (verified against both sources):

``rotate_iou`` boxes are ``[x, y, w, h, angle]`` and ``rbbox_to_corners`` rotates
the local corners by ``R(-angle)``::

    x' =  cos(a) * x + sin(a) * y + cx
    y' = -sin(a) * x + cos(a) * y + cy

``iou3d_nms`` boxes are ``[x, y, z, dx, dy, dz, heading]`` and
``rotate_around_center`` rotates by ``R(+heading)``::

    x' = dx * cos(h) - dy * sin(h) + cx
    y' = dx * sin(h) + dy * cos(h) + cy

Both build local corners with the third column spanning x and the fourth
spanning y, so the equivalent 7-column box is ``[x, y, 0, w, h, 1, -angle]``.
"""

import numpy as np
import torch

from ...ops.iou3d_nms import iou3d_nms_cuda


def _to_boxes7(boxes):
    """Convert ``[x, y, w, h, angle]`` rows to ``[x, y, z, dx, dy, dz, heading]``."""
    boxes = np.asarray(boxes, dtype=np.float32)
    out = np.zeros((boxes.shape[0], 7), dtype=np.float32)
    out[:, 0] = boxes[:, 0]
    out[:, 1] = boxes[:, 1]
    out[:, 3] = boxes[:, 2]
    out[:, 4] = boxes[:, 3]
    out[:, 5] = 1.0
    # Opposite rotation sign; see the module docstring.
    out[:, 6] = -boxes[:, 4]
    return out


def _overlap_area(boxes7, query7):
    """Rotated-box intersection area, matching ``rotate_iou.inter``."""
    n, k = boxes7.shape[0], query7.shape[0]
    if torch.cuda.is_available():
        a = torch.from_numpy(boxes7).cuda().contiguous()
        b = torch.from_numpy(query7).cuda().contiguous()
        overlap = torch.cuda.FloatTensor(torch.Size((n, k))).zero_()
        iou3d_nms_cuda.boxes_overlap_bev_gpu(a, b, overlap)
        return overlap.cpu().numpy()

    # CPU fallback: the extension only exposes IoU on CPU, so invert
    # iou = I / (A + B - I)  ->  I = iou * (A + B) / (1 + iou).
    a = torch.from_numpy(boxes7).contiguous()
    b = torch.from_numpy(query7).contiguous()
    iou = torch.zeros((n, k), dtype=torch.float32)
    iou3d_nms_cuda.boxes_iou_bev_cpu(a, b, iou)
    iou = iou.numpy()
    area_a = (boxes7[:, 3] * boxes7[:, 4]).reshape(n, 1)
    area_b = (query7[:, 3] * query7[:, 4]).reshape(1, k)
    with np.errstate(divide='ignore', invalid='ignore'):
        return iou * (area_a + area_b) / (1.0 + iou)


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """Rotated box IoU with the same signature/semantics as the numba version.

    Args:
        boxes: ``(N, 5)`` float array of ``[x, y, w, h, angle]``.
        query_boxes: ``(K, 5)`` float array of ``[x, y, w, h, angle]``.
        criterion: ``-1`` union IoU, ``0`` divide by ``boxes`` area, ``1`` divide
            by ``query_boxes`` area, anything else returns the raw
            intersection area.
        device_id: accepted for signature compatibility and ignored.

    Returns:
        ``(N, K)`` array in the dtype of ``boxes``.
    """
    box_dtype = boxes.dtype
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)
    n, k = boxes.shape[0], query_boxes.shape[0]
    if n == 0 or k == 0:
        return np.zeros((n, k), dtype=box_dtype)

    inter = _overlap_area(_to_boxes7(boxes), _to_boxes7(query_boxes))

    area1 = (boxes[:, 2] * boxes[:, 3]).reshape(n, 1)
    area2 = (query_boxes[:, 2] * query_boxes[:, 3]).reshape(1, k)
    # The numba kernel divides unguarded; keep the same behaviour for
    # degenerate zero-area boxes rather than silently changing AP.
    with np.errstate(divide='ignore', invalid='ignore'):
        if criterion == -1:
            out = inter / (area1 + area2 - inter)
        elif criterion == 0:
            out = inter / area1
        elif criterion == 1:
            out = inter / area2
        else:
            out = inter
    return np.ascontiguousarray(out, dtype=np.float32).astype(box_dtype)
