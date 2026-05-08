import random
import re
import signal
import sys
from pathlib import Path
from typing import Annotated, Optional

import multiprocessing as mp
import numpy as np

from utils.torch_cpp_log import ensure_before_torch_import

ensure_before_torch_import()
import torch
from munch import munchify

import dtyper as typer

from gui import gui_utils, slam_gui
from submodules.dust3r import Dust3r

from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd

sys.path.append("gaussian_splatting")
from scene.gaussian_model import GaussianModel

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def _close_mp_queue(q):
    if q is None or isinstance(q, FakeQueue):
        return
    try:
        q.close()
        q.join_thread()
    except Exception:
        pass


def _flatten_triplet_after_option(argv: list[str], opt: str) -> list[str]:
    """Collapse `opt A B C` into `opt A:B:C` when the next three args are ints."""
    out = []
    i = 0
    n = len(argv)
    while i < n:
        if argv[i] == opt and i + 4 <= n:
            cand = argv[i + 1 : i + 4]
            ok = False
            try:
                for c in cand:
                    int(c, 10)
                ok = True
            except ValueError:
                pass
            if ok:
                out.extend([opt, ":".join(cand)])
                i += 4
                continue
        out.append(argv[i])
        i += 1
    return out


def _parse_range_str(s: Optional[str]) -> Optional[tuple[int, int, int]]:
    if s is None:
        return None
    raw = re.split(r"[\s,:/]+", s.strip())
    parts = [p for p in raw if p != ""]
    if len(parts) != 3:
        raise typer.BadParameter("Expected three integers BEGIN END STEP (subset [BEGIN, END) ).")
    b, e, st = map(int, parts)
    if st < 1:
        raise typer.BadParameter("STEP must be >= 1.")
    if b < 0 or e <= b:
        raise typer.BadParameter("Need 0 <= BEGIN < END (half-open END).")
    return b, e, st


def _apply_dataset_frame_overrides(cfg: dict, triple: tuple[int, int, int]) -> None:
    ds = cfg.setdefault("Dataset", {})
    b, e, st = triple
    ds["frame_begin"] = b
    ds["frame_end"] = e
    ds["frame_step"] = st


def _check_frame_span(cfg: dict, n_ds: int) -> None:
    ds = cfg.get("Dataset") or {}
    begin = max(0, int(ds.get("frame_begin", 0)))
    fe_raw = ds.get("frame_end")
    end_exc = min(n_ds, n_ds if fe_raw is None else int(fe_raw))
    if begin >= n_ds or begin >= end_exc:
        raise ValueError(f"Invalid frame subset: begin={begin}, end_exclusive={end_exc}, n={n_ds}")


class SLAM:
    """Loads config, constructs map model and multiprocessing pipelines; call run() to execute SLAM."""

    def __init__(self, config):
        self.device = "cuda"
        self.dtype = torch.float32

        self.config = config
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        self.model_params, self.opt_params = (model_params, opt_params)

        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_gui = self.config["Results"]["use_gui"]

        self.dataset = load_dataset(model_params, model_params.source_path, config=config)
        assert len(self.dataset) >= 2

        _check_frame_span(self.config, len(self.dataset))

        raw_w, raw_h = self.dataset.width, self.dataset.height
        raw_K = self.dataset.K
        dust3r = Dust3r(K=raw_K, w=raw_w, h=raw_h, size=512)

        sh_degree = config["model_params"]["sh_degree"]
        initial_opacity = config["model_params"]["initial_opacity"]

        self.gaussians = GaussianModel(sh_degree, initial_opacity)
        self.gaussians.training_setup(self.opt_params)
        self.background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.dust3r = dust3r
        self.frontend.set_params()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.set_params()

        # Queues are closed in run() after the pipeline finishes.
        self.frontend_queue = frontend_queue
        self.backend_queue = backend_queue
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main

        self.gui_process = None
        self.params_gui = None
        if self.use_gui:
            gui_bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
            dm = float(self.config["Training"]["depth_max_threshold"])
            self.params_gui = gui_utils.ParamsGUI(
                background=gui_bg,
                gaussians=self.gaussians,
                q_main2vis=q_main2vis,
                q_vis2main=q_vis2main,
                depth_vis_min=0.1,
                depth_vis_max=dm * 0.8,
            )

    def shutdown_gui(self):
        if self.gui_process and self.gui_process.is_alive():
            Log("Shutting down viewer GUI...", tag="GUI")
            self.gui_process.terminate()
            self.gui_process.join(timeout=5)
            if self.gui_process.is_alive():
                self.gui_process.kill()
            self.gui_process.close()

    def signal_handler(self, sig, frame):
        self.shutdown_gui()
        sys.exit(0)

    def run(self) -> None:
        """Start backend (and optional GUI), run frontend tracking until finish, then join and clean up."""
        signal.signal(signal.SIGINT, self.signal_handler)

        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui and self.params_gui is not None:
            self.gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            self.gui_process.start()
            import time as _t

            _t.sleep(3)

        backend_process.start()
        self.frontend.run()

        self.backend_queue.put(["pause"])
        self.backend_queue.put(["stop"])
        backend_process.join(timeout=120)
        if backend_process.is_alive():
            Log("Backend did not exit after stop; terminating.", tag="SLAM")
            backend_process.terminate()
            backend_process.join(timeout=30)
        if backend_process.is_alive():
            backend_process.kill()
            backend_process.join(timeout=15)
        backend_process.close()

        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            self.shutdown_gui()

        for q in (
            self.frontend_queue,
            self.backend_queue,
            self.q_main2vis,
            self.q_vis2main,
        ):
            _close_mp_queue(q)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    import os as _os

    _os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


app = typer.Typer(invoke_without_command=True, no_args_is_help=True, add_completion=False, context_settings={"help_option_names": ["-h", "--help"]})


@app.callback(invoke_without_command=True)
def main(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    visualize: Annotated[bool, typer.Option("-v", "--visualize", help="Open GUI")] = False,
    spark_live: Annotated[bool, typer.Option("-w", "--spark-live", help="Enable web visualization of GS using Spark")] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "-l",
            "--verbose",
            help="Verbose console output.",
        ),
    ] = False,
    refine: Annotated[
        Optional[int],
        typer.Option(
            "-r",
            "--refine",
            metavar="ITERS",
            help="Enable map refinement after SLAM with the given number of iterations (overrides YAML).",
        ),
    ] = None,
    range_: Annotated[
        Optional[str],
        typer.Option(
            "--range",
            metavar="BEGIN END STEP",
            help="Half-open [BEGIN, END) with STEP; overrides YAML (e.g. --range 0 1500 2 or --range 0:1500:2).",
        ),
    ] = None,
) -> None:
    rng = _parse_range_str(range_)
    mp.set_start_method("spawn")

    cfg = load_config(str(config))

    if visualize:
        cfg.setdefault("Results", {})["use_gui"] = True
    if spark_live:
        cfg.setdefault("Results", {})["spark_live_enable"] = True
    cfg.setdefault("Results", {})["verbose"] = verbose
    if refine is not None:
        cfg.setdefault("Results", {})["map_refine"] = True
        cfg["Results"]["map_refine_iterations"] = refine
    if rng is not None:
        _apply_dataset_frame_overrides(cfg, rng)

    seed_everything(42)
    SLAM(cfg).run()
    Log("Done.")


if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + _flatten_triplet_after_option(sys.argv[1:], "--range")
    app()
