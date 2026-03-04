#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Evaluate distance estimation quality in CSV files (multi-thread capable).

Key features:
- Auto-detect whether `estimated_distance` is already on the same scale as
  `actual_distance` or is approximately the SQUARED distance (choose between
  identity vs sqrt transform by minimizing median absolute percentage error
  after optimal multiplicative scaling).
- Compute error metrics after optimal multiplicative scaling (zero-intercept LS).
- Single-file or directory mode (with optional recursive scan).
- Multi-threaded directory processing via --workers (files are processed in parallel).
- NEW: --by-type now also works in directory/recursive mode (per-file breakdown rows).

Expected CSV columns:
    query_id, data_id, estimated_distance, actual_distance, type
'''

from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd


def _read_one_csv(path: Path,
                  est_col: str = "estimated_distance",
                  act_col: str = "actual_distance",
                  type_col: str = "type") -> pd.DataFrame:
    usecols = [est_col, act_col, type_col]
    # be robust if 'type' is missing
    try:
        df = pd.read_csv(path, usecols=usecols)
    except ValueError:
        df = pd.read_csv(path)
        if type_col not in df.columns:
            df[type_col] = "all"
        missing = [c for c in (est_col, act_col) if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        df = df[[est_col, act_col, type_col]]

    # keep numeric & positive actuals; estimated must be >0 (zero/neg ignored)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[est_col, act_col])
    df = df[df[act_col] > 0]
    df = df[df[est_col] > 0]
    return df


def _optimal_scale(x: np.ndarray, y: np.ndarray) -> float:
    '''Zero-intercept least squares scale: minimize ||a*x - y||^2.'''
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return 1.0
    return float(np.dot(x, y) / denom)


def _abs_perc_err(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Relative error (absolute), safe for y>0 assumed by filtering
    return np.abs((pred - y) / y)


def _r2_score(pred: np.ndarray, y: np.ndarray) -> float:
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _pick_transform(est: np.ndarray, act: np.ndarray) -> Tuple[str, np.ndarray, float]:
    '''
    Decide between identity vs sqrt transform by comparing
    median absolute percentage error after optimal scaling.
    Returns (transform_name, est_used, scale).
    '''
    # Candidate 1: identity
    x1 = est.copy()
    a1 = _optimal_scale(x1, act)
    pred1 = a1 * x1
    mape1 = float(np.median(_abs_perc_err(pred1, act)))

    # Candidate 2: sqrt (we already filtered est>0; sqrt is valid)
    x2 = np.sqrt(est)
    a2 = _optimal_scale(x2, act)
    pred2 = a2 * x2
    mape2 = float(np.median(_abs_perc_err(pred2, act)))

    if mape2 < mape1:
        return "sqrt", x2, a2
    else:
        return "identity", x1, a1


def _metrics_for_arrays(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    err = pred - y
    are = np.abs(err / y)  # absolute relative error

    metrics = {
        "n": int(y.size),
        "R2": _r2_score(pred, y),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE(%)": float(np.mean(are) * 100.0),
        "MedAPE(%)": float(np.median(are) * 100.0),
        "P50 Abs%(%)": float(np.percentile(are, 50) * 100.0),
        "P90 Abs%(%)": float(np.percentile(are, 90) * 100.0),
        "P95 Abs%(%)": float(np.percentile(are, 95) * 100.0),
        "P99 Abs%(%)": float(np.percentile(are, 99) * 100.0),
        "Bias Mean Rel(%)": float(np.mean(err / y) * 100.0),
        "Within 1%": float(np.mean(are <= 0.01)),
        "Within 5%": float(np.mean(are <= 0.05)),
        "Within 10%": float(np.mean(are <= 0.10)),
        "Within 20%": float(np.mean(are <= 0.20)),
        "Within 50%": float(np.mean(are <= 0.50)),
    }
    return metrics


def evaluate_dataframe(df: pd.DataFrame,
                       est_col: str,
                       act_col: str,
                       type_col: str,
                       by_type: bool = False) -> pd.DataFrame:
    '''Compute metrics (overall and optionally by type).'''
    rows: List[Dict[str, float]] = []

    def _eval_one(sub: pd.DataFrame, label: str) -> Dict[str, float]:
        est = sub[est_col].to_numpy(dtype=float)
        act = sub[act_col].to_numpy(dtype=float)

        transform, x_used, scale = _pick_transform(est, act)
        pred = scale * x_used

        m = _metrics_for_arrays(pred, act)
        m["group"] = label
        m["transform"] = transform
        m["scale"] = float(scale)
        return m

    # overall
    rows.append(_eval_one(df, "ALL"))

    if by_type and (type_col in df.columns):
        for t, sub in df.groupby(type_col):
            if len(sub):
                rows.append(_eval_one(sub, f"type={t}"))

    out = pd.DataFrame(rows)
    # Reorder columns for readability
    order = ["group", "n", "transform", "scale", "R2",
             "MAE", "RMSE",
             "MAPE(%)", "MedAPE(%)",
             "P50 Abs%(%)", "P90 Abs%(%)", "P95 Abs%(%)", "P99 Abs%(%)",
             "Bias Mean Rel(%)",
             "Within 1%", "Within 5%", "Within 10%", "Within 20%", "Within 50%"]
    return out[order]


def _process_file(f: Path,
                  est_col: str,
                  act_col: str,
                  type_col: str,
                  by_type: bool) -> List[Dict[str, object]]:
    '''Worker: read, evaluate, return list of metrics dicts (ALL + optional per-type), each includes file.'''
    df = _read_one_csv(f, est_col, act_col, type_col)
    mdf = evaluate_dataframe(df, est_col, act_col, type_col, by_type=by_type)
    if mdf.empty:
        return [{
            "file": str(f), "group": "EMPTY", "n": 0,
            "transform": "", "scale": float("nan"), "R2": float("nan"),
            "MAE": float("nan"), "RMSE": float("nan"),
            "MAPE(%)": float("nan"), "MedAPE(%)": float("nan"),
            "P50 Abs%(%)": float("nan"), "P90 Abs%(%)": float("nan"),
            "P95 Abs%(%)": float("nan"), "P99 Abs%(%)": float("nan"),
            "Bias Mean Rel(%)": float("nan"),
            "Within 1%": float("nan"), "Within 5%": float("nan"),
            "Within 10%": float("nan"), "Within 20%": float("nan"), "Within 50%": float("nan")
        }]
    rows = mdf.to_dict(orient="records")
    for r in rows:
        r["file"] = str(f)
    return rows


def evaluate_path(path: Path,
                  recursive: bool = False,
                  by_type: bool = False,
                  workers: int = 1,
                  est_col: str = "estimated_distance",
                  act_col: str = "actual_distance",
                  type_col: str = "type") -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    '''
    If path is file -> return (this_file_metrics, None)
    If path is dir  -> return (aggregate listing rows per file, including per-type when by_type=True)
    In directory mode, files are processed in parallel when workers > 1.
    '''
    if path.is_file():
        df = _read_one_csv(path, est_col, act_col, type_col)
        metrics = evaluate_dataframe(df, est_col, act_col, type_col, by_type=by_type)
        return metrics, None

    if path.is_dir():
        pattern = "**/*.csv" if recursive else "*.csv"
        files = sorted(path.glob(pattern))
        rows: List[Dict[str, object]] = []

        if not files:
            return pd.DataFrame(), None

        if workers is None or workers <= 1:
            # sequential
            for f in files:
                try:
                    rows.extend(_process_file(f, est_col, act_col, type_col, by_type))
                except Exception as e:
                    rows.append({
                        "file": str(f), "group": "ERROR", "n": 0,
                        "transform": "", "scale": np.nan, "R2": np.nan,
                        "MAE": np.nan, "RMSE": np.nan,
                        "MAPE(%)": np.nan, "MedAPE(%)": np.nan,
                        "P50 Abs%(%)": np.nan, "P90 Abs%(%)": np.nan,
                        "P95 Abs%(%)": np.nan, "P99 Abs%(%)": np.nan,
                        "Bias Mean Rel(%)": np.nan,
                        "Within 1%": np.nan, "Within 5%": np.nan,
                        "Within 10%": np.nan, "Within 20%": np.nan, "Within 50%": np.nan,
                        "error": str(e),
                    })
        else:
            # parallel
            max_workers = workers if workers > 0 else min(8, (os.cpu_count() or 4))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                future2file = {
                    ex.submit(_process_file, f, est_col, act_col, type_col, by_type): f for f in files
                }
                for fut in as_completed(future2file):
                    f = future2file[fut]
                    try:
                        rows.extend(fut.result())
                    except Exception as e:
                        rows.append({
                            "file": str(f), "group": "ERROR", "n": 0,
                            "transform": "", "scale": np.nan, "R2": np.nan,
                            "MAE": np.nan, "RMSE": np.nan,
                            "MAPE(%)": np.nan, "MedAPE(%)": np.nan,
                            "P50 Abs%(%)": np.nan, "P90 Abs%(%)": np.nan,
                            "P95 Abs%(%)": np.nan, "P99 Abs%(%)": np.nan,
                            "Bias Mean Rel(%)": np.nan,
                            "Within 1%": np.nan, "Within 5%": np.nan,
                            "Within 10%": np.nan, "Within 20%": np.nan, "Within 50%": np.nan,
                            "error": str(e),
                        })

        if rows:
            cols_order = ["file", "group", "n", "transform", "scale", "R2",
                          "MAE", "RMSE", "MAPE(%)", "MedAPE(%)",
                          "P50 Abs%(%)", "P90 Abs%(%)", "P95 Abs%(%)", "P99 Abs%(%)",
                          "Bias Mean Rel(%)",
                          "Within 1%", "Within 5%", "Within 10%", "Within 20%", "Within 50%",
                          "error"]
            summary = pd.DataFrame(rows)
            # Keep requested order if columns present
            ordered = [c for c in cols_order if c in summary.columns]
            summary = summary[ordered + [c for c in summary.columns if c not in ordered]]
        else:
            summary = pd.DataFrame()
        return summary, None

    raise FileNotFoundError(f"Path not found: {path}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate distance estimation quality in CSVs (multi-thread capable).")
    ap.add_argument("path", type=str, help="CSV file or a directory containing CSVs.")
    ap.add_argument("--recursive", action="store_true", help="If path is a directory, recursively scan for CSVs.")
    ap.add_argument("--by-type", action="store_true", help="Show per-type breakdown (works in both single-file and directory modes).")
    ap.add_argument("--out", type=str, default=None, help="If set: save summary/metrics to this CSV file.")
    ap.add_argument("--est-col", type=str, default="estimated_distance", help="Estimated distance column name.")
    ap.add_argument("--act-col", type=str, default="actual_distance", help="Actual distance column name.")
    ap.add_argument("--type-col", type=str, default="type", help="Type column name (for grouping).")
    ap.add_argument("--workers", type=int, default=0,
                    help="Number of worker threads for directory mode (0 -> auto, 1 -> sequential).")

    args = ap.parse_args()

    p = Path(args.path)
    workers = (min(72, os.cpu_count() or 4) if args.workers == 0 else args.workers)

    table, _ = evaluate_path(
        p,
        recursive=args.recursive,
        by_type=args.by_type,
        workers=workers,
        est_col=args.est_col,
        act_col=args.act_col,
        type_col=args.type_col
    )

    if table is None or table.empty:
        print("No results.")
        return

    # Print nicely
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(table.to_string(index=False))

    # Save if requested
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
