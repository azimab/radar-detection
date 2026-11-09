#!/usr/bin/env python3
"""AV-24 rig geometry, read from the car's URDF instead of transcribed.

Every mounting offset this pipeline needs -- where the radar sits, where the CoG
sits, where the GNSS antennas sit -- is already declared once, in

    third_party/av24_urdf/av24.urdf    (a pinned copy; AV24_URDF overrides it)

which is the file the car itself publishes ``/tf_static`` from. Transcribing those
numbers into a python constant is how the rear-radar pipeline and the lidar pipeline
end up disagreeing about the same car, so this module parses the URDF and answers
frame questions from it. ``pp_kitti_common`` keeps hand-copied versions of two of them
(``COG_IN_RAM`` and ``RAM_Z_ABOVE_GROUND``); :func:`check_pp_kitti_agrees` asserts the
copies still match rather than trusting them.

Only fixed joints are read. The AV-24 URDF has nothing else, and a revolute joint
would make "the" transform a function of a joint angle we do not have -- so an
unexpected joint type raises instead of being silently treated as fixed.

    >>> rig = Av24Rig.load()
    >>> rig.translation("rear_axle_middle", "radar_rear").round(4)
    array([-0.498,  0.   ,  0.233])
    >>> round(math.degrees(rig.rpy("rear_axle_middle", "radar_rear")[1]), 3)
    -5.0
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_URDF = Path(
    os.environ.get("AV24_URDF", Path(__file__).resolve().parent / "third_party" / "av24_urdf" / "av24.urdf")
)


def _rpy_matrix(rpy: Iterable[float]) -> np.ndarray:
    """URDF ``rpy`` -> rotation matrix.

    URDF fixes the axes: R = Rz(yaw) @ Ry(pitch) @ Rx(roll), which is what scipy
    spells as the *extrinsic* (lowercase) "xyz" sequence. The intrinsic "XYZ" one
    composes in the opposite order and silently disagrees whenever more than one of
    the three angles is non-zero -- as it is for the AV-24 rear radar.
    """
    return Rotation.from_euler("xyz", list(rpy)).as_matrix()


def _pose(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _rpy_matrix(rpy)
    matrix[:3, 3] = np.asarray(list(xyz), dtype=np.float64)
    return matrix


def _floats(text: str | None, default: tuple[float, float, float]) -> tuple[float, ...]:
    if text is None:
        return default
    values = tuple(float(part) for part in text.split())
    if len(values) != 3:
        raise ValueError(f"expected three numbers in URDF attribute, got {text!r}")
    return values


@dataclass(frozen=True)
class Av24Rig:
    """The AV-24 link tree: ``child -> (parent, T_parent_child)``."""

    edges: dict[str, tuple[str, np.ndarray]]
    source: Path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_URDF) -> "Av24Rig":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"AV-24 URDF not found at {path}; set AV24_URDF to the car's av24.urdf "
                "or unset it to use third_party/av24_urdf/av24.urdf"
            )
        root = ET.parse(path).getroot()
        edges: dict[str, tuple[str, np.ndarray]] = {}
        for joint in root.findall("joint"):
            name = joint.get("name", "<unnamed>")
            kind = joint.get("type")
            if kind != "fixed":
                raise ValueError(
                    f"{path}: joint {name!r} is {kind!r}, not 'fixed'; its transform "
                    "depends on a joint angle this pipeline does not have"
                )
            child = joint.find("child").get("link")
            parent = joint.find("parent").get("link")
            origin = joint.find("origin")
            xyz = _floats(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
            rpy = _floats(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
            if child in edges:
                raise ValueError(f"{path}: link {child!r} has more than one parent joint")
            edges[child] = (parent, _pose(xyz, rpy))
        if not edges:
            raise ValueError(f"{path}: no fixed joints found")
        return cls(edges=edges, source=path)

    @property
    def links(self) -> set[str]:
        return set(self.edges) | {parent for parent, _ in self.edges.values()}

    def _chain_to_root(self, link: str) -> list[str]:
        chain, current, seen = [link], link, {link}
        while current in self.edges:
            current = self.edges[current][0]
            if current in seen:
                raise ValueError(f"cycle in URDF above {link!r}")
            seen.add(current)
            chain.append(current)
        return chain

    def _to_root(self, link: str) -> np.ndarray:
        """Pose of ``link`` in its tree's root frame."""
        if link not in self.links:
            raise KeyError(f"{link!r} is not a link in {self.source}")
        matrix = np.eye(4, dtype=np.float64)
        current = link
        while current in self.edges:
            parent, t_parent_current = self.edges[current]
            matrix = t_parent_current @ matrix
            current = parent
        return matrix

    def transform(self, target: str, source: str) -> np.ndarray:
        """``T_target_source``: the pose of ``source`` expressed in ``target``.

        Applied to a column vector it takes coordinates in ``source`` to ``target``,
        the same direction as ROS's ``lookup_transform(target, source)``.
        """
        root_t, root_s = self._chain_to_root(target)[-1], self._chain_to_root(source)[-1]
        if root_t != root_s:
            raise ValueError(
                f"{target!r} and {source!r} are in disconnected trees "
                f"(roots {root_t!r} and {root_s!r}) in {self.source}"
            )
        t_root_target = self._to_root(target)
        inverse = np.eye(4, dtype=np.float64)
        inverse[:3, :3] = t_root_target[:3, :3].T
        inverse[:3, 3] = -t_root_target[:3, :3].T @ t_root_target[:3, 3]
        return inverse @ self._to_root(source)

    def translation(self, target: str, source: str) -> np.ndarray:
        return self.transform(target, source)[:3, 3]

    def rpy(self, target: str, source: str) -> np.ndarray:
        """Roll/pitch/yaw in radians, in the same fixed-axis convention as the URDF."""
        return Rotation.from_matrix(self.transform(target, source)[:3, :3]).as_euler("xyz")


# --- frames this pipeline names ------------------------------------------------------
#
# The rear radar reports in ``radar_rear``; boxes, points and the Doppler compensation
# all live in ``rear_axle_middle``; the ego twist is reported at ``center_of_gravity``.
RAM = "rear_axle_middle"
COG = "center_of_gravity"
RADAR_REAR = "radar_rear"
GROUND = "rear_axle_middle_ground"
# The two Novatel antennas straddle the centreline by +-0.075 m. An RTK fix solved
# from the pair lands on their midpoint, which is not a URDF link, so it is built from
# the pair rather than picked arbitrarily.
NOVATEL_PAIR = ("novatel_a", "novatel_b")


def novatel_midpoint_in(rig: Av24Rig, target: str = RAM) -> np.ndarray:
    """Midpoint of the Novatel antenna pair, expressed in ``target``."""
    a, b = (rig.translation(target, link) for link in NOVATEL_PAIR)
    return 0.5 * (a + b)


# --- what the URDF can and cannot say about the car's outline -------------------------
#
# The URDF declares no body geometry (its only visual is a mesh that is not available
# here), so the AV-24's extents cannot be read off it. What it does give is hard
# points the body must contain: the GNSS nose antenna is mounted on the nose, so the
# front of the car reaches at least that far forward, and the rear radar is mounted in
# the rear bodywork, so the back reaches at least that far back. Those two turn a known
# car length into a window the box centre has to fall inside.
NOSE_LANDMARK = "front_antenna"
TAIL_LANDMARK = RADAR_REAR


def box_center_bounds_in_ram(rig: Av24Rig, car_length: float) -> tuple[float, float]:
    """Window the AV-24 box centre must fall in, along x in ``rear_axle_middle``.

    ``pp_kitti_common`` puts the opponent's box centre at ``CAR_L / 2`` ahead of its
    rear_axle_middle, which is right only if rear_axle_middle sits on the rear bumper.
    The URDF says it does not -- ``radar_rear`` is already 0.498 m behind it -- so that
    placement is structurally too far forward.

    Bounding the centre is not measuring it. This cannot supply the right value; it can
    only reject one that no AV-24 could produce.
    """
    half = 0.5 * float(car_length)
    nose = float(rig.translation(RAM, NOSE_LANDMARK)[0])
    tail = float(rig.translation(RAM, TAIL_LANDMARK)[0])
    low, high = nose - half, tail + half  # centre far enough fwd / far enough back
    if low > high:
        raise ValueError(
            f"car_length {car_length} m is too short to span {TAIL_LANDMARK} at "
            f"{tail:.4f} m and {NOSE_LANDMARK} at {nose:.4f} m in {rig.source}"
        )
    return low, high


def pp_box_center_in_ram(pp: object) -> float:
    """Where ``pp_kitti_common`` places the opponent box centre, along x in RAM."""
    return float(np.asarray(getattr(pp, "OPP_BOX_CENTER_IN_RAM"), dtype=np.float64)[0])


def check_pp_kitti_agrees(rig: Av24Rig, pp: object, tolerance: float = 1e-6) -> None:
    """Fail if ``pp_kitti_common``'s transcribed rig constants have drifted from the URDF.

    The lidar pipeline hardcodes two numbers this module can derive: ``COG_IN_RAM``,
    which every opponent box is built on, and ``RAM_Z_ABOVE_GROUND``, the vertical datum
    every box floor rests on. Both are the same physical quantities as the URDF's
    ``rear_axle_middle -> center_of_gravity`` and ``rear_axle_middle_ground ->
    rear_axle_middle``. The radar and lidar datasets are only comparable while they stay
    that way, so this is checked rather than assumed.
    """
    urdf_cog = rig.translation(RAM, COG)
    pp_cog = np.asarray(getattr(pp, "COG_IN_RAM"), dtype=np.float64)
    if not np.allclose(urdf_cog, pp_cog, atol=tolerance):
        raise ValueError(
            f"pp_kitti_common.COG_IN_RAM {pp_cog.tolist()} disagrees with "
            f"{rig.source} rear_axle_middle -> center_of_gravity {urdf_cog.tolist()}"
        )
    # Sign: the URDF hangs rear_axle_middle *above* the ground link, so RAM expressed in
    # the ground frame has the positive height pp_kitti_common calls RAM_Z_ABOVE_GROUND.
    urdf_ride_height = float(rig.translation(GROUND, RAM)[2])
    pp_ride_height = float(getattr(pp, "RAM_Z_ABOVE_GROUND"))
    if abs(urdf_ride_height - pp_ride_height) > tolerance:
        raise ValueError(
            f"pp_kitti_common.RAM_Z_ABOVE_GROUND {pp_ride_height} disagrees with "
            f"{rig.source} rear_axle_middle_ground -> rear_axle_middle z "
            f"{urdf_ride_height}"
        )


def _main() -> None:
    rig = Av24Rig.load()
    print(f"{rig.source}\n")
    print(f"{'link':<28}{'x':>9}{'y':>9}{'z':>9}{'roll':>9}{'pitch':>9}{'yaw':>9}   (in rear_axle_middle)")
    for link in sorted(rig.links - {RAM}):
        try:
            xyz = rig.translation(RAM, link)
            rpy = np.degrees(rig.rpy(RAM, link))
        except ValueError:
            continue
        print(
            f"{link:<28}{xyz[0]:>9.4f}{xyz[1]:>9.4f}{xyz[2]:>9.4f}"
            f"{rpy[0]:>9.3f}{rpy[1]:>9.3f}{rpy[2]:>9.3f}"
        )
    mid = novatel_midpoint_in(rig)
    print(f"\n{'novatel midpoint':<28}{mid[0]:>9.4f}{mid[1]:>9.4f}{mid[2]:>9.4f}")
    print(f"{'novatel midpoint - CoG':<28}" + "".join(
        f"{v:>9.4f}" for v in mid - rig.translation(RAM, COG)
    ))
    low, high = box_center_bounds_in_ram(rig, 4.876)
    print(f"\nbox centre admissible window for a 4.876 m car: "
          f"[{low:+.4f}, {high:+.4f}] m ahead of {RAM}")


if __name__ == "__main__":
    _main()
