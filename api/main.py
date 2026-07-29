"""Serve XDecomposer through a small FastAPI application.

Run from the repository root::

    uvicorn api.main:app --host 0.0.0.0 --port 8000

The checkpoints and materials database can be overridden with the environment
variables XDECOMPOSER_CHECKPOINT, XDECOMPOSER_MAE_CHECKPOINT and
XDECOMPOSER_MATERIALS_DB.
"""

from __future__ import annotations

import logging
import os
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any

import numpy as np
import torch
import torch.nn.functional as F
from ase.db import connect
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field, model_validator
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.cif import CifWriter
from pymatgen.io.ase import AseAtomsAdaptor

from src.models.xdecomposer import XDecomposer, build_xdecomposer
from src.models.xrd_transformer import XRDTransformerEncoder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / " checkpoints" / "sepration" / "latest.pt"
DEFAULT_MAE_CHECKPOINT = ROOT / " checkpoints" / "pretrain" / "best_model.pt"
DEFAULT_MATERIALS_DB = ROOT / "MP500.db"
DEFAULT_REFERENCE_CACHE = ROOT / ".cache" / "mp500_reference_bank.pt"
DEFAULT_MAX_REFERENCES = 500
DEFAULT_XRDBENCH_H5 = ROOT / "datasets" / "mp500" / "patterns.h5"
# Uvicorn configures this logger at INFO level, so startup progress is visible
# in the same terminal that runs `uvicorn`.
logger = logging.getLogger("uvicorn.error")


class DecomposeRequest(BaseModel):
    """One measured XRD pattern.

    `intensities` may have any length.  It is interpolated to the 3,500 points
    expected by the trained model.  If `two_theta` is omitted, the values are
    assumed to be uniformly sampled from 10 to 80 degrees.
    """

    intensities: list[float] = Field(min_length=2, description="Measured XRD intensities.")
    two_theta: list[float] | None = Field(
        default=None, description="2θ values in degrees, one for each intensity."
    )

    @model_validator(mode="after")
    def validate_two_theta(self) -> "DecomposeRequest":
        if self.two_theta is not None and len(self.two_theta) != len(self.intensities):
            raise ValueError("two_theta and intensities must have the same length")
        return self


class PhaseResult(BaseModel):
    phase_index: int
    active: bool
    activity_probability: float
    mp_id: str
    match_score: float
    structure_url: str
    alternatives: list[dict[str, float | str]]
    xrd: list[float] | None = None


class DecomposeResponse(BaseModel):
    xrd_length: int
    two_theta_range: tuple[float, float]
    phases: list[PhaseResult]


@dataclass
class ReferenceBank:
    patterns: torch.Tensor
    mp_ids: list[str]
    row_ids: list[int]


class DecompositionService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: XDecomposer | None = None
        self.reference_bank: ReferenceBank | None = None
        self.xrd_length = 3500
        self.materials_db_path: Path | None = None

    def load(self) -> None:
        checkpoint_path = Path(os.getenv("XDECOMPOSER_CHECKPOINT", DEFAULT_CHECKPOINT))
        mae_path = Path(os.getenv("XDECOMPOSER_MAE_CHECKPOINT", DEFAULT_MAE_CHECKPOINT))
        db_path = Path(os.getenv("XDECOMPOSER_MATERIALS_DB", DEFAULT_MATERIALS_DB))
        for path in (checkpoint_path, mae_path, db_path):
            if not path.is_file():
                raise RuntimeError(f"Required model resource not found: {path}")

        logger.info("Loading XDecomposer checkpoint on %s...", self.device)

        # mmap avoids keeping both 250+ MB checkpoints resident while the model
        # is being constructed.  This matters on CPU-only deployment machines.
        mae_checkpoint = torch.load(mae_path, map_location="cpu", weights_only=False, mmap=True)
        mae_config: dict[str, Any] = mae_checkpoint.get("config", {})
        del mae_checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        config: dict[str, Any] = checkpoint["config"]
        self.xrd_length = int(config["xrd_length"])
        # build_xdecomposer only needs the MAE encoder.  Avoid instantiating
        # the unused MAE decoder (~100 MB) in a production API process.
        mae = SimpleNamespace(
            d_model=mae_config.get("d_model", 768),
            xrd_length=self.xrd_length,
            use_rope=False,
            encoder=XRDTransformerEncoder(
                n_layers=mae_config.get("n_layers", 4),
                d_model=mae_config.get("d_model", 768),
                n_heads=mae_config.get("n_heads", 12),
                d_ff=1024,
                dropout=0.1,
            ),
        )
        model = build_xdecomposer(
            mae,
            num_sources=config["num_phases"],
            cnn_channels=config.get("cnn_channels", [64, 128, 256, 512]),
            cnn_kernels=config.get("cnn_kernels"),
            cnn_strides=config.get("cnn_strides"),
            use_transformer=not config.get("no_transformer", False),
            use_film=not config.get("no_film", False),
            use_skip_connections=not config.get("no_skip_connections", False),
            mask_type=config.get("mask_type", "soft"),
        )
        # XDecomposer deep-copies only the temporary MAE encoder.
        del mae
        state = {key.removeprefix("module."): value for key, value in checkpoint["model_state_dict"].items()}
        model.load_state_dict(state)
        del state, checkpoint
        self.model = model.to(self.device).eval()
        self.materials_db_path = db_path
        logger.info("Model loaded. Preparing MP500 reference bank...")
        self.reference_bank = self._build_reference_bank(db_path)
        logger.info("API startup complete: %d reference structures are ready.", len(self.reference_bank.mp_ids))

    def _build_reference_bank(self, db_path: Path) -> ReferenceBank:
        """Load cached references or calculate a reference XRD per MP500 structure."""
        cache_path = Path(os.getenv("XDECOMPOSER_REFERENCE_CACHE", DEFAULT_REFERENCE_CACHE))
        db_mtime_ns = db_path.stat().st_mtime_ns
        try:
            max_references = int(os.getenv("XDECOMPOSER_MAX_REFERENCES", str(DEFAULT_MAX_REFERENCES)))
        except ValueError as exc:
            raise RuntimeError("XDECOMPOSER_MAX_REFERENCES must be a positive integer") from exc
        if max_references < 1:
            raise RuntimeError("XDECOMPOSER_MAX_REFERENCES must be a positive integer")
        if cache_path.is_file():
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                if (
                    cached["xrd_length"] == self.xrd_length
                    and cached["db_mtime_ns"] == db_mtime_ns
                    and cached["max_references"] == max_references
                ):
                    logger.info("Loading cached reference bank from %s...", cache_path)
                    tensor = cached["patterns"].to(self.device)
                    return ReferenceBank(
                        patterns=F.normalize(tensor, p=2, dim=1),
                        mp_ids=cached["mp_ids"],
                        row_ids=cached["row_ids"],
                    )
                logger.info("Reference cache is outdated; rebuilding it.")
            except (KeyError, RuntimeError, EOFError):
                logger.warning("Reference cache is unreadable; rebuilding it.")

        logger.info("Building a %d-structure reference bank from %s...", max_references, db_path)
        grid = np.linspace(10.0, 80.0, self.xrd_length, dtype=np.float32)
        calculator = XRDCalculator()
        adaptor = AseAtomsAdaptor()
        patterns: list[np.ndarray] = []
        mp_ids: list[str] = []
        row_ids: list[int] = []
        db = connect(str(db_path))
        total = min(db.count(), max_references)
        for row_index, row in enumerate(db.select(limit=max_references), start=1):
            mp_id = str(row.key_value_pairs.get("mpid", "")).removesuffix(".cif")
            if not mp_id:
                continue
            try:
                diffraction = calculator.get_pattern(adaptor.get_structure(row.toatoms()), two_theta_range=(10, 80))
                # Convert discrete Bragg peaks to a small Gaussian instrumental broadening.
                signal = np.zeros_like(grid)
                for position, intensity in zip(diffraction.x, diffraction.y):
                    signal += float(intensity) * np.exp(-0.5 * ((grid - position) / 0.12) ** 2)
                maximum = signal.max()
                if maximum <= 0:
                    continue
                patterns.append(signal / maximum)
                mp_ids.append(mp_id)
                row_ids.append(row.id)
            except Exception:
                # A malformed structure must not prevent the service from starting.
                pass
            if row_index % 25 == 0 or row_index == total:
                logger.info("Reference bank progress: %d/%d structures", row_index, total)
        if not patterns:
            raise RuntimeError(f"No usable structures found in materials database: {db_path}")
        tensor_cpu = torch.from_numpy(np.stack(patterns))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "xrd_length": self.xrd_length,
                "db_mtime_ns": db_mtime_ns,
                "max_references": max_references,
                "patterns": tensor_cpu,
                "mp_ids": mp_ids,
                "row_ids": row_ids,
            },
            cache_path,
        )
        logger.info("Reference bank cached at %s", cache_path)
        tensor = tensor_cpu.to(self.device)
        return ReferenceBank(patterns=F.normalize(tensor, p=2, dim=1), mp_ids=mp_ids, row_ids=row_ids)

    def structure_cif(self, mp_id: str) -> str:
        """Return the matched structure as a downloadable CIF document."""
        if self.reference_bank is None or self.materials_db_path is None:
            raise RuntimeError("The reference bank is not loaded")
        try:
            row_id = self.reference_bank.row_ids[self.reference_bank.mp_ids.index(mp_id)]
        except ValueError as exc:
            raise KeyError(mp_id) from exc
        row = connect(str(self.materials_db_path)).get(id=row_id)
        structure = AseAtomsAdaptor().get_structure(row.toatoms())
        return str(CifWriter(structure))

    def theoretical_peaks(self, mp_id: str) -> dict[str, Any]:
        """Calculate Cu Kα Bragg peaks for a matched candidate structure."""
        if self.reference_bank is None or self.materials_db_path is None:
            raise RuntimeError("The reference bank is not loaded")
        try:
            row_id = self.reference_bank.row_ids[self.reference_bank.mp_ids.index(mp_id)]
        except ValueError as exc:
            raise KeyError(mp_id) from exc
        row = connect(str(self.materials_db_path)).get(id=row_id)
        structure = AseAtomsAdaptor().get_structure(row.toatoms())
        pattern = XRDCalculator().get_pattern(structure, two_theta_range=(10, 80))
        return {
            "mp_id": mp_id,
            "radiation": "Cu Kα",
            "two_theta": [float(value) for value in pattern.x],
            "intensities": [float(value) for value in pattern.y],
        }

    def decompose(self, request: DecomposeRequest, top_k: int, include_patterns: bool) -> DecomposeResponse:
        if self.model is None or self.reference_bank is None:
            raise RuntimeError("The model is not loaded")
        intensity = np.asarray(request.intensities, dtype=np.float32)
        if not np.isfinite(intensity).all():
            raise ValueError("intensities must be finite numbers")
        if request.two_theta is None:
            angles = np.linspace(10.0, 80.0, len(intensity), dtype=np.float32)
        else:
            angles = np.asarray(request.two_theta, dtype=np.float32)
            if not np.isfinite(angles).all() or np.any(np.diff(angles) <= 0):
                raise ValueError("two_theta must contain finite, strictly increasing values")
        target_angles = np.linspace(10.0, 80.0, self.xrd_length, dtype=np.float32)
        # np.interp always returns float64, while the PyTorch checkpoint uses
        # float32 parameters.  Cast explicitly before inference.
        signal = np.interp(target_angles, angles, intensity, left=0.0, right=0.0).astype(np.float32)
        signal = np.clip(signal, 0.0, None)
        maximum = signal.max()
        if maximum <= 0:
            raise ValueError("intensities must contain at least one positive value")
        input_tensor = torch.from_numpy(signal / maximum).to(self.device).view(1, 1, -1)
        with torch.inference_mode():
            separated, logits = self.model(input_tensor)
            phase_patterns = separated[0]
            activity = torch.sigmoid(logits[0])
            similarities = F.normalize(phase_patterns, p=2, dim=1) @ self.reference_bank.patterns.T
            scores, indices = torch.topk(similarities, k=min(top_k, len(self.reference_bank.mp_ids)), dim=1)

        results: list[PhaseResult] = []
        for phase_index in range(phase_patterns.shape[0]):
            alternatives = [
                {"mp_id": self.reference_bank.mp_ids[int(index)], "match_score": float(score)}
                for score, index in zip(scores[phase_index].cpu(), indices[phase_index].cpu())
            ]
            probability = float(activity[phase_index].cpu())
            results.append(PhaseResult(
                phase_index=phase_index,
                active=probability >= 0.5,
                activity_probability=probability,
                mp_id=alternatives[0]["mp_id"],
                match_score=alternatives[0]["match_score"],
                structure_url=f"/v1/structures/{alternatives[0]['mp_id']}.cif",
                alternatives=alternatives,
                xrd=phase_patterns[phase_index].cpu().tolist() if include_patterns else None,
            ))
        return DecomposeResponse(xrd_length=self.xrd_length, two_theta_range=(10.0, 80.0), phases=results)


service = DecompositionService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.load()
    yield


app = FastAPI(title="XDecomposer API", version="1.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def interface() -> FileResponse:
    """Human-facing XRD upload and result-export interface."""
    return FileResponse(ROOT / "api" / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if service.model is not None else "starting"}


@app.get("/v1/examples/xrdbench/multiphase/{index}")
def xrdbench_multiphase_example(index: int) -> dict[str, Any]:
    """Return one locally installed XRDbench multiphase sample for the web UI."""
    dataset_path = Path(os.getenv("XRDBENCH_PATTERNS_H5", DEFAULT_XRDBENCH_H5))
    if not dataset_path.is_file():
        raise HTTPException(status_code=404, detail=f"XRDbench data not found: {dataset_path}")
    if index < 0:
        raise HTTPException(status_code=422, detail="Example index must be non-negative")
    try:
        import h5py

        with h5py.File(dataset_path, "r") as handle:
            total = len(handle["multi_intensity"])
            if index >= total:
                raise HTTPException(status_code=404, detail=f"Example index must be between 0 and {total - 1}")
            raw_ids = handle["multi_phase_ids_json"][index]
            if isinstance(raw_ids, bytes):
                raw_ids = raw_ids.decode("utf-8")
            return {
                "name": f"XRDbench multiphase #{index}",
                "two_theta": handle["two_theta_deg"][:].tolist(),
                "intensities": handle["multi_intensity"][index].tolist(),
                "ground_truth": [
                    {"mp_id": mp_id, "fraction": float(fraction)}
                    for mp_id, fraction in zip(json.loads(raw_ids), handle["multi_phase_fractions"][index])
                ],
            }
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read XRDbench data: {exc}") from exc


@app.get("/v1/structures/{mp_id}.cif", response_class=PlainTextResponse)
def structure_cif(mp_id: str) -> PlainTextResponse:
    """Download the CIF of a structure currently present in the reference bank."""
    try:
        cif = service.structure_cif(mp_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"mp-id is not in the loaded reference bank: {mp_id}") from exc
    return PlainTextResponse(
        cif,
        media_type="chemical/x-cif",
        headers={"Content-Disposition": f'attachment; filename="{mp_id}.cif"'},
    )


@app.get("/v1/structures/{mp_id}/theoretical-peaks")
def theoretical_peaks(mp_id: str) -> dict[str, Any]:
    """Return theoretical Bragg-peak sticks for comparison with an experimental XRD."""
    try:
        return service.theoretical_peaks(mp_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"mp-id is not in the loaded reference bank: {mp_id}") from exc


@app.post("/v1/decompose", response_model=DecomposeResponse)
def decompose(
    request: DecomposeRequest,
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
    include_patterns: bool = False,
) -> DecomposeResponse:
    """Decompose an XRD mixture and retrieve the nearest MP structures."""
    try:
        return service.decompose(request, top_k=top_k, include_patterns=include_patterns)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
