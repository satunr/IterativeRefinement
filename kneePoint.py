from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator, find_shape
from scipy.interpolate import UnivariateSpline


CurveType = Literal["concave", "convex"]
DirectionType = Literal["increasing", "decreasing"]


@dataclass
class KneePointResult:
	knee_x: float
	knee_y: float
	knee_index: int
	f: Callable[[float | np.ndarray], np.ndarray]
	x: np.ndarray
	y: np.ndarray
	x_normalized: np.ndarray
	y_normalized: np.ndarray
	difference_curve: np.ndarray
	curve: CurveType
	direction: DirectionType


def _prepare_xy(x: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
	x_arr = np.asarray(list(x), dtype=float)
	y_arr = np.asarray(list(y), dtype=float)

	if x_arr.size != y_arr.size:
		raise ValueError("x and y must have the same number of elements")
	if x_arr.size < 3:
		raise ValueError("Need at least 3 points to detect a knee")

	order = np.argsort(x_arr)
	x_sorted = x_arr[order]
	y_sorted = y_arr[order]

	x_unique, unique_idx = np.unique(x_sorted, return_index=True)
	y_unique = y_sorted[unique_idx]

	if x_unique.size < 3:
		raise ValueError("Need at least 3 distinct x values")

	return x_unique, y_unique


def _auto_select_mode(y: np.ndarray) -> tuple[CurveType, DirectionType]:
	# Backward-compatible fallback used when find_shape cannot be evaluated.
	direction: DirectionType = "increasing" if y[-1] >= y[0] else "decreasing"
	second_diff = np.diff(y, n=2)
	curve: CurveType = "concave" if second_diff.size == 0 or float(np.nanmean(second_diff)) < 0 else "convex"
	return curve, direction


def _max_distance_knee_location(x_arr: np.ndarray, y_arr: np.ndarray, dense_points: int = 2000, 
								aggressiveness: float = 1.0,) -> tuple[int, float]:
	# Fit a smooth spline and estimate the knee as the maximum distance
	# from the secant line joining the endpoints.
	degree = min(3, len(x_arr) - 1)
	spline = UnivariateSpline(x_arr, y_arr, k=degree, s=0)
	x_dense = np.linspace(float(x_arr[0]), float(x_arr[-1]), dense_points)
	y_dense = np.asarray(spline(x_dense), dtype=float)
	x0, y0 = float(x_dense[0]), float(y_dense[0])
	x1, y1 = float(x_dense[-1]), float(y_dense[-1])
	denominator = float(np.hypot(x1 - x0, y1 - y0))
	if np.isclose(denominator, 0.0):
		knee_dense_idx = dense_points // 2
	else:
		distances = np.abs((y1 - y0) * x_dense - (x1 - x0) * y_dense + x1 * y0 - y1 * x0) / denominator
		x_norm = (x_dense - x_dense.min()) / (x_dense.max() - x_dense.min())
		score = distances * np.exp(aggressiveness * (1.0 - x_norm))
		knee_dense_idx = int(np.nanargmax(score))
	knee_x = float(x_dense[knee_dense_idx])
	knee_idx = int(np.argmin(np.abs(x_arr - knee_x)))
	return knee_idx, knee_x


def detect_knee_point(x: Iterable[float], y: Iterable[float], curve: Optional[CurveType] = None, 
					  direction: Optional[DirectionType] = None, aggressiveness: float = 1.0,) -> KneePointResult:
	"""
	Detect a knee point using smoothed endpoint distance, with kneed diagnostics.

	Parameters
	----------
	x, y:
		Input graph points.
	curve:
		"concave" or "convex". If None, inferred from data.
	direction:
		"increasing" or "decreasing". If None, inferred from endpoints.
	aggressiveness:
		Higher values bias knee selection earlier on x (more aggressive). Change this to shift the knee point left (higher values) 
		or right (lower values).

	Returns
	-------
	KneePointResult
		Contains knee coordinates (x, y), function f(x), and diagnostic curves.
	"""
	x_arr, y_arr = _prepare_xy(x, y)

	try:
		auto_direction, auto_curve = find_shape(x_arr, y_arr)
		auto_curve = auto_curve if auto_curve in ("concave", "convex") else _auto_select_mode(y_arr)[0]
		auto_direction = auto_direction if auto_direction in ("increasing", "decreasing") else _auto_select_mode(y_arr)[1]
	except Exception:
		auto_curve, auto_direction = _auto_select_mode(y_arr)

	curve = curve or auto_curve
	direction = direction or auto_direction

	locator = KneeLocator(
		x_arr,
		y_arr,
		curve=curve,
		direction=direction,
	)

	x_norm = np.asarray(locator.x_normalized, dtype=float)
	y_norm = np.asarray(locator.y_normalized, dtype=float)
	diff_curve = np.asarray(locator.y_difference, dtype=float)

	knee_idx, knee_x = _max_distance_knee_location(
		x_arr,
		y_arr,
		dense_points=2000,
		aggressiveness=aggressiveness,
	)
	knee_y = float(np.interp(knee_x, x_arr, y_arr))

	def f(x_query: float | np.ndarray) -> np.ndarray:
		x_query_arr = np.asarray(x_query, dtype=float)
		return np.interp(x_query_arr, x_arr, y_arr)

	return KneePointResult(knee_x=knee_x, knee_y=knee_y, knee_index=knee_idx, f=f, x=x_arr, y=y_arr, x_normalized=x_norm, 
						   y_normalized=y_norm, difference_curve=diff_curve, curve=curve, direction=direction,)


def plot_knee(result: KneePointResult, title: str = "Kneedle Knee Point", function_label: Optional[str] = None,):
	x_dense = np.linspace(result.x.min(), result.x.max(), 500)
	y_dense = result.f(x_dense)

	fig, ax = plt.subplots(figsize=(8, 5))
	curve_label = f"f(x) = {function_label}" if function_label else None
	ax.plot(x_dense, y_dense, "-", color="blue", label=curve_label, linewidth=2)
	ax.scatter(
		[result.knee_x],
		[result.knee_y],
		s=200,
		marker="x",  # type: ignore[arg-type]
		color="red",
		label=f"knee ({result.knee_x:.4g}, {result.knee_y:.4g})",
		zorder=5,
	)

	ax.set_title(title)
	ax.set_xlabel("Edge weight")
	ax.set_ylabel("Fraction of edges kept")
	ax.grid(alpha=0.25)
	ax.legend()
	fig.tight_layout()
	output_path = Path(__file__).resolve().parent / "knee_point_plot.png"
	plt.savefig(output_path, dpi=300)
	plt.show()


'''
if __name__ == "__main__":

	x_ex = np.linspace(0, 20, 300)
	y_ex = np.exp(-2 * x_ex)

	result = detect_knee_point(x_ex, y_ex)

	print("Detected mode:", f"{result.direction} {result.curve} curve,")
	print("Knee point (x, y):", (f"{result.knee_x:.4f}", f"{result.knee_y:.4f}"))

	plot_knee(result, title="Kneedle Knee Detection", function_label=f"exp(-2x)")
'''
