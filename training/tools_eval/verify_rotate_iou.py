"""Compare the torch-backed rotate_iou_gpu_eval against a pure-numpy transcription
of the numba rbbox_to_corners + convex-polygon-intersection algorithm."""
import numpy as np


def rbbox_to_corners_ref(rbbox):
    """Exact transcription of rotate_iou.rbbox_to_corners."""
    cx, cy, x_d, y_d, angle = rbbox
    a_cos, a_sin = np.cos(angle), np.sin(angle)
    corners_x = np.array([-x_d / 2, -x_d / 2, x_d / 2, x_d / 2])
    corners_y = np.array([-y_d / 2, y_d / 2, y_d / 2, -y_d / 2])
    xs = a_cos * corners_x + a_sin * corners_y + cx
    ys = -a_sin * corners_x + a_cos * corners_y + cy
    return np.stack([xs, ys], axis=1)


def poly_clip(subject, clip):
    """Sutherland-Hodgman clip of convex `subject` by convex `clip`."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    # orient clip counter-clockwise
    def signed_area(poly):
        s = 0.0
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            s += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
        return s / 2.0

    if signed_area(clip) < 0:
        clip = clip[::-1]
    out = [tuple(p) for p in subject]
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        if not out:
            break
        inp, out = out, []
        for j in range(len(inp)):
            cur, prv = inp[j], inp[j - 1]
            d_cur, d_prv = inside(cur, a, b), inside(prv, a, b)
            if d_cur >= 0:
                if d_prv < 0:
                    t = d_prv / (d_prv - d_cur)
                    out.append((prv[0] + t * (cur[0] - prv[0]),
                                prv[1] + t * (cur[1] - prv[1])))
                out.append(cur)
            elif d_prv >= 0:
                t = d_prv / (d_prv - d_cur)
                out.append((prv[0] + t * (cur[0] - prv[0]),
                            prv[1] + t * (cur[1] - prv[1])))
    return out


def poly_area(poly):
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        s += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(s) / 2.0


def rotate_iou_reference(boxes, query_boxes, criterion=-1):
    n, k = boxes.shape[0], query_boxes.shape[0]
    out = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        c1 = rbbox_to_corners_ref(boxes[i])
        a1 = boxes[i, 2] * boxes[i, 3]
        for j in range(k):
            c2 = rbbox_to_corners_ref(query_boxes[j])
            a2 = query_boxes[j, 2] * query_boxes[j, 3]
            inter = poly_area(poly_clip(c1, c2))
            if criterion == -1:
                out[i, j] = inter / (a1 + a2 - inter)
            elif criterion == 0:
                out[i, j] = inter / a1
            elif criterion == 1:
                out[i, j] = inter / a2
            else:
                out[i, j] = inter
    return out


def main():
    import torch

    from pcdet.datasets.radar.rotate_iou_torch import (
        rotate_iou_gpu_eval,
    )

    on_gpu = torch.cuda.is_available()
    print("backend path:", "GPU boxes_overlap_bev_gpu (exact area)" if on_gpu
          else "CPU boxes_iou_bev_cpu + algebraic inversion")

    rng = np.random.RandomState(0)

    # Car-sized boxes at rear-radar ranges, query boxes biased to overlap so IoU
    # is not trivially zero. Pool every pair: per-trial means would be dominated
    # by single near-tangent outliers in 1-25 pair trials.
    n = k = 120
    boxes = np.zeros((n, 5), dtype=np.float32)
    boxes[:, 0] = rng.uniform(-100, 0, n)
    boxes[:, 1] = rng.uniform(-50, 50, n)
    boxes[:, 2] = rng.uniform(1.0, 6.0, n)
    boxes[:, 3] = rng.uniform(1.0, 3.0, n)
    boxes[:, 4] = rng.uniform(-np.pi, np.pi, n)
    query = boxes.copy()
    query[:, 0] += rng.uniform(-3, 3, k)
    query[:, 1] += rng.uniform(-3, 3, k)
    query[:, 4] = rng.uniform(-np.pi, np.pi, k)

    def wrong_sign(bx, qb, crit):
        """The naive heading=+angle mapping, for a sensitivity baseline."""
        bx, qb = bx.copy(), qb.copy()
        bx[:, 4] *= -1
        qb[:, 4] *= -1
        return rotate_iou_gpu_eval(bx, qb, crit).astype(np.float64)

    names = {-1: "union IoU", 0: "/area(boxes)", 1: "/area(query)",
             2: "raw intersection area"}
    print(f"\nPooled over {n * k} box pairs "
          f"(error vs. a numpy transcription of the numba algorithm):")
    print(f"  {'criterion':<28} {'mean':>10} {'max':>10} {'frac>1e-2':>10}")
    means, sign_means = {}, {}
    # The numba kernel computes `inter` once and only varies the divisor, so the
    # reference does the same: one polygon-clip pass, four derived criteria.
    inter_ref = rotate_iou_reference(boxes, query, 2)
    area_b = (boxes[:, 2] * boxes[:, 3]).astype(np.float64).reshape(n, 1)
    area_q = (query[:, 2] * query[:, 3]).astype(np.float64).reshape(1, k)
    ref_for = {
        -1: inter_ref / (area_b + area_q - inter_ref),
        0: inter_ref / area_b,
        1: inter_ref / area_q,
        2: inter_ref,
    }
    for crit in (-1, 0, 1, 2):
        exp = ref_for[crit]
        got = rotate_iou_gpu_eval(boxes, query, crit).astype(np.float64)
        err = np.abs(got - exp)
        means[crit] = err.mean()
        sign_means[crit] = np.abs(wrong_sign(boxes, query, crit) - exp).mean()
        print(f"  {crit:>2} {names[crit]:<25} {err.mean():10.3e} "
              f"{err.max():10.3e} {np.mean(err > 1e-2):10.3e}")

    print("\n  Same statistic with the WRONG heading sign (sensitivity check):")
    for crit in (-1, 0, 1, 2):
        ratio = sign_means[crit] / max(means[crit], 1e-300)
        print(f"  {crit:>2} {names[crit]:<25} {sign_means[crit]:10.3e}"
              f"   ({ratio:.0f}x worse than the mapping in use)")

    # The correct mapping agrees at float32 noise; the wrong one is orders of
    # magnitude worse. Require both: small absolute error, and a large margin
    # over the sign-flipped variant.
    assert max(means.values()) < 1e-5, "MISMATCH - angle/dim convention is wrong"
    assert min(sign_means[c] / max(means[c], 1e-300) for c in means) > 100, \
        "test is not sensitive to the heading convention"
    print("\nPASS: torch backend matches the numba algorithm.")


if __name__ == "__main__":
    main()
