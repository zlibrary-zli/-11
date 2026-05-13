#!/usr/bin/env python3
import json
import os
import platform
import queue
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / ".imgdiff_gui_state.json"


def quote_arg(arg: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([arg])
    return shlex.quote(arg)


def open_path(path: str) -> None:
    if not path:
        return
    if platform.system() == "Windows":
        os.startfile(path)  # type: ignore[attr-defined]
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class ImgDiffGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("imgdiff Python GUI")
        self.root.geometry("1040x820")
        self.root.minsize(920, 700)

        self.log_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.proc: Optional[subprocess.Popen] = None
        self.last_program = ""
        self.last_args: list[str] = []
        self.last_report_path = ""

        self.vars: dict[str, tk.Variable] = {}
        self._build_vars()
        self._build_ui()
        self._load_state()
        self.root.after(100, self._poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_vars(self) -> None:
        self.vars["mode"] = tk.StringVar(value="single")
        self.vars["single_ref"] = tk.StringVar()
        self.vars["single_tgt"] = tk.StringVar()
        self.vars["single_out"] = tk.StringVar(value=str(ROOT / "out"))
        self.vars["single_prefix"] = tk.StringVar()

        self.vars["dir_ref"] = tk.StringVar()
        self.vars["dir_tgt"] = tk.StringVar()
        self.vars["dir_out"] = tk.StringVar(value=str(ROOT / "out"))

        self.vars["motion"] = tk.StringVar(value="euclidean")
        self.vars["max_dim"] = tk.IntVar(value=2000)
        self.vars["ecc_iters"] = tk.IntVar(value=200)
        self.vars["ecc_eps"] = tk.DoubleVar(value=1e-6)
        self.vars["k_mad"] = tk.DoubleVar(value=6.0)
        self.vars["min_area"] = tk.IntVar(value=30)
        self.vars["open_k"] = tk.IntVar(value=3)
        self.vars["close_k"] = tk.IntVar(value=5)
        self.vars["alpha"] = tk.DoubleVar(value=0.6)

        self.vars["save_overlay"] = tk.BooleanVar(value=True)
        self.vars["save_mask"] = tk.BooleanVar(value=True)
        self.vars["save_regions"] = tk.BooleanVar(value=True)
        self.vars["save_report"] = tk.BooleanVar(value=True)
        self.vars["save_aligned"] = tk.BooleanVar(value=True)

    def _build_ui(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        title = ttk.Label(outer, text="imgdiff", font=("TkDefaultFont", 18, "bold"))
        title.pack(anchor="w")
        subtitle = ttk.Label(
            outer,
            text="Python 启动器：无需 Qt，优先复用/自动构建 C++ 核心，再用图形界面无脑操作。",
        )
        subtitle.pack(anchor="w", pady=(4, 12))

        self.status_var = tk.StringVar(value="就绪")
        self.summary_var = tk.StringVar(value="")

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="x", expand=False)
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.notebook = notebook

        single_tab = ttk.Frame(notebook, padding=12)
        dir_tab = ttk.Frame(notebook, padding=12)
        notebook.add(single_tab, text="单对图片")
        notebook.add(dir_tab, text="目录批处理")

        self._build_single_tab(single_tab)
        self._build_dir_tab(dir_tab)
        self._build_params_group(outer)
        self._build_action_row(outer)

        ttk.Label(outer, textvariable=self.status_var, foreground="#1d4ed8").pack(anchor="w", pady=(8, 2))
        ttk.Label(outer, textvariable=self.summary_var, wraplength=980).pack(anchor="w", pady=(0, 8))

        log_frame = ttk.LabelFrame(outer, text="运行日志", padding=8)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, height=20, wrap="word")
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set)

    def _build_single_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        self._path_row(parent, 0, "ref 文件", self.vars["single_ref"], False)
        self._path_row(parent, 1, "tgt 文件", self.vars["single_tgt"], False)
        self._path_row(parent, 2, "输出目录", self.vars["single_out"], True)
        ttk.Label(parent, text="前缀（可选）").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=8)
        ttk.Entry(parent, textvariable=self.vars["single_prefix"]).grid(row=3, column=1, sticky="ew", pady=8)

    def _build_dir_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        self._path_row(parent, 0, "ref 目录", self.vars["dir_ref"], True)
        self._path_row(parent, 1, "tgt 目录", self.vars["dir_tgt"], True)
        self._path_row(parent, 2, "输出目录", self.vars["dir_out"], True)
        ttk.Label(parent, text="目录模式按同名 .bmp 文件配对。").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )

    def _path_row(self, parent, row: int, label: str, var: tk.Variable, is_dir: bool) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=8)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=8)
        ttk.Button(
            parent,
            text="选择...",
            command=lambda: self._choose_path(var, is_dir),
        ).grid(row=row, column=2, sticky="ew", padx=(10, 0), pady=8)

    def _build_params_group(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="参数", padding=12)
        frame.pack(fill="x", expand=False, pady=(12, 0))
        for idx in range(4):
            frame.columnconfigure(idx, weight=1)

        self._combo_row(frame, 0, 0, "motion", self.vars["motion"], [
            ("euclidean（旋转+平移）", "euclidean"),
            ("affine（仿射）", "affine"),
            ("translation（仅平移）", "translation"),
        ])
        self._spin_row(frame, 0, 2, "max_dim", self.vars["max_dim"], 64, 20000)
        self._spin_row(frame, 1, 0, "ecc_iters", self.vars["ecc_iters"], 1, 20000)
        self._entry_row(frame, 1, 2, "ecc_eps", self.vars["ecc_eps"])
        self._entry_row(frame, 2, 0, "k_mad", self.vars["k_mad"])
        self._spin_row(frame, 2, 2, "min_area", self.vars["min_area"], 0, 1000000)
        self._spin_row(frame, 3, 0, "open_k", self.vars["open_k"], 0, 255)
        self._spin_row(frame, 3, 2, "close_k", self.vars["close_k"], 0, 255)
        self._entry_row(frame, 4, 0, "alpha", self.vars["alpha"])

        out_frame = ttk.Frame(frame)
        out_frame.grid(row=4, column=2, sticky="ew", padx=(0, 10), pady=8)
        ttk.Label(out_frame, text="输出").pack(side="left", padx=(0, 8))
        for key, text in [
            ("save_overlay", "overlay"),
            ("save_mask", "mask"),
            ("save_regions", "regions.csv"),
            ("save_report", "report.json"),
            ("save_aligned", "tgt_aligned"),
        ]:
            ttk.Checkbutton(out_frame, text=text, variable=self.vars[key]).pack(side="left", padx=(0, 8))

    def _combo_row(self, parent, row: int, col: int, label: str, var: tk.StringVar, items) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=8)
        combo = ttk.Combobox(parent, textvariable=var, state="readonly")
        combo["values"] = [text for text, _ in items]
        combo.current(0)
        combo.grid(row=row, column=col + 1, sticky="ew", pady=8)
        combo.bind("<<ComboboxSelected>>", lambda _e: var.set(dict(items)[combo.get()]))
        var.set(items[0][1])

        def sync_combo(*_args):
            for text, value in items:
                if value == var.get():
                    combo.set(text)
                    break

        var.trace_add("write", sync_combo)
        sync_combo()

    def _spin_row(self, parent, row: int, col: int, label: str, var: tk.IntVar, min_v: int, max_v: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=8)
        ttk.Spinbox(parent, from_=min_v, to=max_v, textvariable=var).grid(row=row, column=col + 1, sticky="ew", pady=8)

    def _entry_row(self, parent, row: int, col: int, label: str, var: tk.Variable) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 10), pady=8)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=col + 1, sticky="ew", pady=8)

    def _build_action_row(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", expand=False, pady=(12, 0))

        self.run_button = ttk.Button(row, text="运行", command=self.run_clicked)
        self.run_button.pack(side="left")

        ttk.Button(row, text="一键构建核心", command=self.build_only_clicked).pack(side="left", padx=(8, 0))
        ttk.Button(row, text="复制命令行", command=self.copy_command).pack(side="left", padx=(8, 0))
        ttk.Button(row, text="打开输出目录", command=self.open_output_dir).pack(side="left", padx=(8, 0))
        ttk.Button(row, text="打开报告", command=self.open_report).pack(side="left", padx=(8, 0))

    def _choose_path(self, var: tk.Variable, is_dir: bool) -> None:
        current = str(var.get()).strip()
        start_dir = current if current else str(ROOT)
        if is_dir:
            path = filedialog.askdirectory(initialdir=start_dir)
        else:
            path = filedialog.askopenfilename(
                initialdir=str(Path(start_dir).parent if Path(start_dir).exists() else ROOT),
                filetypes=[
                    ("Images", "*.bmp *.png *.jpg *.jpeg *.tif *.tiff"),
                    ("All Files", "*.*"),
                ],
            )
        if path:
            var.set(path)

    def _on_tab_changed(self, _event=None) -> None:
        idx = self.notebook.index(self.notebook.select())
        self.vars["mode"].set("single" if idx == 0 else "dir")

    def log(self, text: str) -> None:
        self.log_text.insert("end", text)
        self.log_text.see("end")

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "done":
                    self._on_process_finished(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def find_imgdiff(self) -> str:
        candidates = [
            ROOT / "build" / "imgdiff",
            ROOT / "build" / "imgdiff.exe",
            ROOT / "imgdiff",
            ROOT / "imgdiff.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        from_path = shutil_which("imgdiff")
        return from_path or ""

    def build_command(self):
        return ["cmake", "-S", str(ROOT), "-B", str(ROOT / "build")]

    def build_compile_command(self):
        return ["cmake", "--build", str(ROOT / "build"), "-j"]

    def collect_args(self):
        mode = self.vars["mode"].get()
        args: list[str] = []

        if mode == "single":
            ref_path = str(self.vars["single_ref"].get()).strip()
            tgt_path = str(self.vars["single_tgt"].get()).strip()
            out_dir = str(self.vars["single_out"].get()).strip()
            prefix = str(self.vars["single_prefix"].get()).strip()
            if not ref_path or not tgt_path:
                return "请先选择 ref/tgt 文件。", None
            if not out_dir:
                return "请先选择输出目录。", None
            args += ["--ref", ref_path, "--tgt", tgt_path, "--out", out_dir]
            if prefix:
                args += ["--prefix", prefix]
        else:
            ref_dir = str(self.vars["dir_ref"].get()).strip()
            tgt_dir = str(self.vars["dir_tgt"].get()).strip()
            out_dir = str(self.vars["dir_out"].get()).strip()
            if not ref_dir or not tgt_dir:
                return "请先选择 ref/tgt 目录。", None
            if not out_dir:
                return "请先选择输出目录。", None
            args += ["--ref_dir", ref_dir, "--tgt_dir", tgt_dir, "--out", out_dir]

        args += ["--motion", str(self.vars["motion"].get()).strip()]
        args += ["--max_dim", str(self.vars["max_dim"].get())]
        args += ["--ecc_iters", str(self.vars["ecc_iters"].get())]
        args += ["--ecc_eps", str(self.vars["ecc_eps"].get())]
        args += ["--k_mad", str(self.vars["k_mad"].get())]
        args += ["--min_area", str(self.vars["min_area"].get())]
        args += ["--open_k", str(self.vars["open_k"].get())]
        args += ["--close_k", str(self.vars["close_k"].get())]
        args += ["--alpha", str(self.vars["alpha"].get())]
        args += ["--save_overlay", "1" if self.vars["save_overlay"].get() else "0"]
        args += ["--save_mask", "1" if self.vars["save_mask"].get() else "0"]
        args += ["--save_regions", "1" if self.vars["save_regions"].get() else "0"]
        args += ["--save_report", "1" if self.vars["save_report"].get() else "0"]
        args += ["--save_aligned", "1" if self.vars["save_aligned"].get() else "0"]
        return None, args

    def run_clicked(self) -> None:
        if self.proc is not None:
            return

        err, args = self.collect_args()
        if err:
            messagebox.showwarning("参数缺失", err)
            return

        program = self.find_imgdiff()
        if not program:
            if not messagebox.askyesno("未找到 imgdiff", "没找到 imgdiff 可执行文件，是否现在自动构建？"):
                return
            self._start_background_task(self._build_then_run, args)
            return

        self._start_subprocess(program, args)

    def build_only_clicked(self) -> None:
        if self.proc is not None:
            return
        self._start_background_task(self._build_only_task, None)

    def _start_background_task(self, fn, args) -> None:
        self.run_button.configure(state="disabled")
        self.summary_var.set("")
        self.status_var.set("处理中...")
        self.log_text.delete("1.0", "end")

        def runner():
            try:
                fn(args)
            except Exception as exc:
                self.log_queue.put(("log", f"\n[ERROR] {exc}\n"))
                self.log_queue.put(("done", json.dumps({"ok": False, "message": str(exc)})))

        threading.Thread(target=runner, daemon=True).start()

    def _build_only_task(self, _args) -> None:
        ok, message = self._run_build_steps()
        self.log_queue.put(("done", json.dumps({"ok": ok, "message": message})))

    def _build_then_run(self, args: list[str]) -> None:
        ok, message = self._run_build_steps()
        if not ok:
            self.log_queue.put(("done", json.dumps({"ok": False, "message": message})))
            return
        program = self.find_imgdiff()
        if not program:
            self.log_queue.put(("done", json.dumps({"ok": False, "message": "构建完成，但仍未找到 imgdiff。"})))
            return
        self.log_queue.put(("log", "\n[INFO] 构建完成，开始运行任务。\n"))
        self.log_queue.put(("done", json.dumps({"ok": True, "message": "build_then_run", "program": program, "args": args})))

    def _run_build_steps(self) -> tuple[bool, str]:
        commands = [
            self.build_command(),
            self.build_compile_command(),
        ]
        for cmd in commands:
            pretty = " ".join(quote_arg(x) for x in cmd)
            self.log_queue.put(("log", f"$ {pretty}\n"))
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                self.log_queue.put(("log", line))
            rc = proc.wait()
            if rc != 0:
                return False, f"构建失败，退出码 {rc}"
        return True, "构建成功"

    def _start_subprocess(self, program: str, args: list[str]) -> None:
        self.summary_var.set("")
        self.status_var.set("运行中...")
        self.log_text.delete("1.0", "end")
        self.last_program = program
        self.last_args = list(args)

        cmd_text = " ".join([quote_arg(program), *[quote_arg(a) for a in args]])
        self.log(f"$ {cmd_text}\n")
        self.run_button.configure(state="disabled")

        self.proc = subprocess.Popen(
            [program, *args],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def reader():
            assert self.proc is not None
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.log_queue.put(("log", line))
            rc = self.proc.wait()
            self.log_queue.put(("done", json.dumps({"ok": rc == 0, "message": f"run:{rc}"})))

        threading.Thread(target=reader, daemon=True).start()

    def _on_process_finished(self, payload: str) -> None:
        data = json.loads(payload)
        message = data.get("message", "")

        if message == "build_then_run" and data.get("ok"):
            self._start_subprocess(data["program"], data["args"])
            return

        self.run_button.configure(state="normal")
        self.proc = None

        if data.get("ok"):
            self.status_var.set("完成")
            self._load_report_summary()
            return

        self.status_var.set("失败")
        messagebox.showerror("执行失败", message or "任务执行失败")

    def _current_out_dir(self) -> str:
        if self.vars["mode"].get() == "single":
            return str(self.vars["single_out"].get()).strip()
        return str(self.vars["dir_out"].get()).strip()

    def _single_report_path(self) -> str:
        out_dir = str(self.vars["single_out"].get()).strip()
        ref_path = str(self.vars["single_ref"].get()).strip()
        prefix = str(self.vars["single_prefix"].get()).strip()
        if not out_dir or not ref_path:
            return ""
        name = prefix or Path(ref_path).stem
        return str(Path(out_dir) / f"{name}_report.json")

    def _load_report_summary(self) -> None:
        if self.vars["mode"].get() != "single" or not self.vars["save_report"].get():
            self.summary_var.set("任务已完成。")
            self.last_report_path = ""
            return

        report_path = self._single_report_path()
        self.last_report_path = report_path
        if not report_path or not Path(report_path).exists():
            self.summary_var.set("任务已完成。")
            return

        data = load_json(Path(report_path))
        if not isinstance(data, dict):
            self.summary_var.set(f"任务已完成，报告路径：{report_path}")
            return

        method = data.get("align_method", "")
        thr = data.get("diff_threshold", "")
        pix = data.get("diff_pixels", "")
        comps = data.get("diff_components", "")
        self.summary_var.set(
            f"对齐={method}  阈值={thr}  diff_pixels={pix}  diff_components={comps}  报告={report_path}"
        )

    def copy_command(self) -> None:
        if not self.last_program:
            err, args = self.collect_args()
            if err:
                messagebox.showwarning("无法复制", err)
                return
            program = self.find_imgdiff() or str(ROOT / "build" / "imgdiff")
            self.last_program = program
            self.last_args = args or []

        cmd = " ".join([quote_arg(self.last_program), *[quote_arg(a) for a in self.last_args]])
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.status_var.set("命令行已复制")

    def open_output_dir(self) -> None:
        out_dir = self._current_out_dir()
        if not out_dir:
            messagebox.showwarning("未设置", "请先填写输出目录。")
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        open_path(out_dir)

    def open_report(self) -> None:
        report_path = self.last_report_path or self._single_report_path()
        if not report_path or not Path(report_path).exists():
            messagebox.showwarning("未找到", "当前没有可打开的报告文件。")
            return
        open_path(report_path)

    def _save_state(self) -> None:
        data = {}
        for key, var in self.vars.items():
            data[key] = var.get()
        STATE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_state(self) -> None:
        data = load_json(STATE_PATH)
        if not isinstance(data, dict):
            return
        for key, value in data.items():
            if key not in self.vars:
                continue
            try:
                self.vars[key].set(value)
            except Exception:
                pass
        if self.vars["mode"].get() == "dir":
            self.notebook.select(1)
        else:
            self.notebook.select(0)

    def on_close(self) -> None:
        self._save_state()
        self.root.destroy()


def shutil_which(binary: str) -> str:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    exts = [""]
    if os.name == "nt":
        exts += os.environ.get("PATHEXT", ".EXE").split(os.pathsep)
    for folder in paths:
        if not folder:
            continue
        for ext in exts:
            candidate = Path(folder) / f"{binary}{ext}"
            if candidate.exists():
                return str(candidate)
    return ""


def main() -> int:
    root = tk.Tk()
    ImgDiffGui(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
