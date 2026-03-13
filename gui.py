import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import yaml
import sys
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from main import run_main_pipeline
from src.utils import resolve_path

CONFIG_FILE = "config.yaml"
DEBUG = 0   # set to 1 to enable full traceback in console

# ── Max palette ─────────────────────────────────────────────────────––––––––––
WHITE   = "#FFFFFF"
OFF     = "#F5F5F3"       # alternating row tint
RULE    = "#CCCCCC"       # hairline rules
DARK    = "#1A1A1A"       # near-black text
MID     = "#767676"       # secondary text
RED     = "#E63329"       # Swiss signal red — the single accent
REDHOV  = "#BF2922"
BGMAIN  = "#FAFAFA"

FONT_CAPS  = ("Helvetica", 8, "bold")
FONT_BODY  = ("Helvetica", 11)
FONT_MONO  = ("Courier",   10)
FONT_SMALL = ("Helvetica", 9)
FONT_NUM   = ("Helvetica", 13)


# ── Stdout/stderr redirector — strips ANSI codes and writes to the GUI console ─
import re as _re
_ANSI = _re.compile(r"\x1b\[[0-9;]*m|\x1b\[[0-9;]*[A-Za-z]")

class TextRedirector:
    """Redirects stdout/stderr to a Tkinter Text widget, buffering by line."""
    def __init__(self, widget):
        self.widget  = widget
        self._buf    = ""          # buffer incomplete lines

    def write(self, text):
        text = _ANSI.sub("", text)   # strip ANSI escape codes
        if not text:
            return
        self._buf += text
        # only flush complete lines to avoid half-written epoch rows
        if "\n" in self._buf:
            lines = self._buf.split("\n")
            self._buf = lines[-1]          # keep incomplete tail
            to_write  = "\n".join(lines[:-1]) + "\n"
            self.widget.configure(state="normal")
            if "error" in to_write.lower() or "!!!" in to_write:
                self.widget.insert(tk.END, to_write, "error")
            elif "completed" in to_write.lower() or "rmse" in to_write.lower():
                self.widget.insert(tk.END, to_write, "accent")
            else:
                self.widget.insert(tk.END, to_write, "normal")
            self.widget.see(tk.END)
            self.widget.configure(state="disabled")

    def flush(self):
        pass


# ── Config helpers ────────────────────────────────────────────────────────────

def get_config_path():
    """Returns the path to config.yaml — next to the executable when compiled, cwd in development."""
    if hasattr(sys, "_MEIPASS"):
        exe_dir = os.path.dirname(sys.executable)
        root = os.path.abspath(os.path.join(exe_dir, "../../.."))
    else:
        root = os.path.abspath(os.getcwd())
    return os.path.join(root, "config.yaml")

def load_config():
    """Loads config.yaml. Raises FileNotFoundError with the resolved path if missing."""
    path = get_config_path()
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"config.yaml not found at: {path}")

def save_config(cfg):
    """Writes the current config dict back to config.yaml."""
    with open(get_config_path(), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


# ── Pipeline thread — runs the full pipeline asynchronously ──────────────────
def run_pipeline_thread(dataset_path, forecast_horizon, look_back,
                        n_epochs, target_var, rnn_flag, output_folder,
                        on_start=None, on_done=None):
    if on_start:
        on_start()
    # Silence Keras epoch logs
    import os as _os
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    try:
        import tensorflow as _tf
        _tf.get_logger().setLevel("ERROR")
    except Exception:
        pass
    try:
        import keras as _keras
        _keras.utils.disable_interactive_logging()   # Keras 3
    except Exception:
        pass
    try:
        print(f"\nStarting pipeline\n")
        print(f"  model            {'RNN' if rnn_flag else 'GPR'}")
        print(f"  forecast_horizon {forecast_horizon}")
        print(f"  look_back        {look_back}")
        print(f"  n_epochs         {n_epochs}")
        print(f"  target_var       {target_var}\n")
        rmse = run_main_pipeline(dataset_path, forecast_horizon, look_back,
                                 n_epochs, target_var, rnn_flag, output_folder)
        import time; time.sleep(2);          # placeholder
        print(f"Pipeline completed — RMSE {rmse:.6f}\n")
        print(f"Output saved to {output_folder}\n")
    except Exception as e:
        if DEBUG:
            import traceback
            print(f"\nERROR (debug mode):\n")
            traceback.print_exc()
        else:
            print(f"\nERROR: {e}\n")
    finally:
        if on_done:
            on_done()


# ── Widget helpers ────────────────────────────────────────────────────────────

def hrule(parent, color=RULE, thick=1, padx=0, pady=0):
    """Draws a horizontal rule inside parent."""
    tk.Frame(parent, bg=color, height=thick).pack(fill="x", padx=padx, pady=pady)

def section_heading(parent, text, padx=20, top_pad=20):
    """Red all-caps section label."""
    f = tk.Frame(parent, bg=WHITE)
    f.pack(fill="x", padx=padx, pady=(top_pad, 6))
    tk.Label(f, text=text.upper(), font=FONT_CAPS, bg=WHITE, fg=RED).pack(side="left")

def param_row(parent, label, value, idx):
    """Single parameter row with alternating background. Returns the Entry widget."""
    bg = OFF if idx % 2 == 0 else WHITE
    row = tk.Frame(parent, bg=bg, padx=20, pady=10)
    row.pack(fill="x")
    tk.Label(row, text=label.upper(), font=FONT_CAPS,
             bg=bg, fg=MID, width=22, anchor="w").pack(side="left")
    e = tk.Entry(row, font=FONT_NUM, width=10,
                 bg=bg, fg=DARK, relief="flat", bd=0,
                 insertbackground=RED,
                 highlightthickness=1,
                 highlightbackground=RULE,
                 highlightcolor=RED)
    e.insert(0, value)
    e.pack(side="left")
    return e

def swiss_btn(parent, text, cmd, primary=False, width=24):
    """Styled button with hover effect. primary=True gives a filled red button."""
    bg = RED if primary else WHITE
    fg = WHITE if primary else DARK
    brd = RED if primary else RULE
    b = tk.Button(parent, text=text.upper(), command=cmd,
                  font=FONT_CAPS, bg=bg, fg=fg,
                  relief="flat", bd=0, cursor="hand2",
                  width=width, padx=14, pady=8,
                  activebackground=REDHOV, activeforeground=WHITE,
                  highlightthickness=1, highlightbackground=brd)
    b.bind("<Enter>", lambda e: b.config(bg=REDHOV, fg=WHITE, highlightbackground=REDHOV))
    b.bind("<Leave>", lambda e: b.config(bg=bg, fg=fg, highlightbackground=brd))
    return b


# ── Main GUI ──────────────────────────────────────────────────────────────────
def create_gui():
    cfg = load_config()
    dataset_path  = [None]
    output_folder = [None]

    root = tk.Tk()
    root.title("PREDator")
    root.geometry("1240x840")
    root.configure(bg=WHITE)
    root.resizable(True, True)
    root.grid_columnconfigure(0, weight=0, minsize=360)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(1, weight=1)

    # ── HEADER ───────────────────────────────────────────────────────────────
    header = tk.Frame(root, bg=WHITE)
    header.grid(row=0, column=0, columnspan=2, sticky="ew",
                padx=28, pady=(24, 0))

    # Logo — falls back to a red square if the icon file is missing
    try:
        from PIL import Image, ImageTk as _ITk
        def _res(r):
            import sys, os
            if hasattr(sys, "_MEIPASS"): return os.path.join(sys._MEIPASS, r)
            return os.path.join(os.path.abspath("."), r)
        _hi = Image.open(_res("icona.icns")).resize((44, 44))
        _hp = _ITk.PhotoImage(_hi)
        _hl = tk.Label(header, image=_hp, bg=WHITE)
        _hl.image = _hp
        _hl.pack(side="left", padx=(0, 12))
    except Exception:
        tk.Label(header, text="■", font=("Helvetica", 20),
                 bg=WHITE, fg=RED).pack(side="left", padx=(0, 10))
    title_block = tk.Frame(header, bg=WHITE)
    title_block.pack(side="left")
    tk.Label(title_block, text="PREDator",
             font=("Helvetica", 20, "bold"), bg=WHITE, fg=DARK).pack(anchor="w")
    tk.Label(title_block, text="PREDICTION & FORECAST ENGINE",
             font=FONT_CAPS, bg=WHITE, fg=MID).pack(anchor="w")
    tk.Label(header, text=" v1.0 ", font=FONT_CAPS,
             bg=DARK, fg=WHITE, padx=6, pady=3).pack(side="right")

    # thick rule under header
    tk.Frame(root, bg=DARK, height=2).grid(
        row=0, column=0, columnspan=2, sticky="sew", padx=0, pady=(56, 0))

    # ── LEFT COLUMN ───────────────────────────────────────────────────────────
    left = tk.Frame(root, bg=WHITE)
    left.grid(row=1, column=0, sticky="nsew")
    left.grid_columnconfigure(0, weight=1)
    tk.Frame(root, bg=RULE, width=1).grid(row=1, column=0, sticky="nse")

    # PARAMETERS
    section_heading(left, "Parameters")
    hrule(left)

    param_specs = [
        ("forecast_horizon", "model",   "forecast_horizon"),
        ("look_back",        "model",   "look_back"),
        ("n_epochs",         "model",   "n_epochs"),
        ("target_var",       "dataset", "target_var"),
    ]
    entries  = {}
    RNN_ONLY = ["look_back", "n_epochs"]  # disabled when GPR is selected

    for i, (label, sec, key) in enumerate(param_specs):
        val = cfg.get(sec, {}).get(key, "")
        entries[label] = param_row(left, label, val, i)

    # MODEL TYPE
    section_heading(left, "Model type", top_pad=16)
    hrule(left)

    model_var = tk.StringVar(value="RNN" if cfg.get("flags", {}).get("rnn_flag", 1) == 1 else "GPR")

    def on_model_change(*_):
        """Disables RNN-only fields when GPR is selected."""
        is_rnn = model_var.get() == "RNN"
        for lbl in RNN_ONLY:
            e = entries[lbl]
            if is_rnn:
                e.config(state="normal", fg=DARK, highlightbackground=RULE)
            else:
                e.config(state="disabled", fg=RULE, highlightbackground=RULE,
                         disabledforeground=RULE, disabledbackground=OFF)

    model_var.trace_add("write", on_model_change)

    rb_frame = tk.Frame(left, bg=WHITE, padx=20, pady=12)
    rb_frame.pack(fill="x")
    for opt in ("RNN", "GPR"):
        tk.Radiobutton(rb_frame, text=opt, variable=model_var, value=opt,
                       font=("Helvetica", 11, "bold"),
                       bg=WHITE, fg=DARK, selectcolor=WHITE,
                       activebackground=WHITE, activeforeground=RED,
                       cursor="hand2", relief="flat").pack(side="left", padx=(0, 24))

    on_model_change()

    # DATA PATHS
    section_heading(left, "Data paths", top_pad=16)
    hrule(left)

    def path_row(btn_text, cmd, idx):
        """File/folder selector row. Returns the label showing the selected path."""
        bg = OFF if idx % 2 == 0 else WHITE
        f = tk.Frame(left, bg=bg, padx=20, pady=10)
        f.pack(fill="x")
        b = tk.Button(f, text=btn_text.upper(), command=cmd,
                      font=FONT_CAPS, bg=WHITE, fg=DARK,
                      relief="flat", bd=0, cursor="hand2",
                      padx=8, pady=4,
                      highlightthickness=1, highlightbackground=RULE,
                      activebackground=RED, activeforeground=WHITE)
        b.pack(side="left")
        lbl = tk.Label(f, text="—", font=FONT_SMALL,
                       bg=bg, fg=MID, anchor="w", wraplength=200)
        lbl.pack(side="left", padx=12)
        return lbl

    def sel_dataset():
        p = filedialog.askopenfilename(title="Select CSV dataset",
                                       filetypes=[("CSV files", "*.csv")])
        if p:
            dataset_path[0] = p
            ds_lbl.config(text=os.path.basename(p), fg=DARK)
            print(f"Dataset: {p}")

    def sel_output():
        f = filedialog.askdirectory(title="Select output folder")
        if f:
            output_folder[0] = f
            out_lbl.config(text=os.path.basename(f), fg=RED)
            print(f"Output folder: {f}")

    ds_lbl  = path_row("Select dataset",       sel_dataset, 0)
    out_lbl = path_row("Select output folder", sel_output,  1)

    # EXECUTE
    section_heading(left, "Execute", top_pad=16)
    hrule(left)

    status_var = tk.StringVar(value="IDLE")
    st_frame = tk.Frame(left, bg=WHITE, padx=20, pady=8)
    st_frame.pack(fill="x")
    tk.Label(st_frame, text="STATUS", font=FONT_CAPS,
             bg=WHITE, fg=MID).pack(side="left")
    status_lbl = tk.Label(st_frame, textvariable=status_var,
                          font=("Helvetica", 9, "bold"), bg=WHITE, fg=MID)
    status_lbl.pack(side="left", padx=10)

    # Indeterminate progress bar — active during pipeline execution
    style = ttk.Style()
    style.theme_use("default")
    style.configure("swiss.Horizontal.TProgressbar",
                    troughcolor=OFF, background=RED,
                    darkcolor=RED, lightcolor=RED,
                    bordercolor=WHITE, thickness=3)
    pbar = ttk.Progressbar(left, style="swiss.Horizontal.TProgressbar",
                           mode="indeterminate")
    pbar.pack(fill="x", padx=20, pady=(0, 10))

    def on_start():
        status_var.set("RUNNING")
        status_lbl.config(fg=RED)
        pbar.start(10)

    def on_done():
        pbar.stop()
        status_var.set("IDLE")
        status_lbl.config(fg=MID)

    def save_and_run():
        """Validates inputs, saves config, and launches the pipeline in a daemon thread."""
        if not dataset_path[0]:
            messagebox.showerror("Error", "Select a CSV dataset first.")
            return
        if not output_folder[0]:
            messagebox.showerror("Error", "Select an output folder first.")
            return
        try:
            cfg["model"]["forecast_horizon"] = int(entries["forecast_horizon"].get())
            cfg["model"]["look_back"]        = int(entries["look_back"].get())
            cfg["model"]["n_epochs"]         = int(entries["n_epochs"].get())
            cfg["dataset"]["target_var"]     = entries["target_var"].get()
            cfg["flags"]["rnn_flag"]         = 1 if model_var.get() == "RNN" else 0
            save_config(cfg)
            print("Config saved.\n")
            threading.Thread(
                target=run_pipeline_thread,
                args=(dataset_path[0],
                      cfg["model"]["forecast_horizon"],
                      cfg["model"]["look_back"],
                      cfg["model"]["n_epochs"],
                      cfg["dataset"]["target_var"],
                      cfg["flags"]["rnn_flag"],
                      output_folder[0],
                      on_start, on_done),
                daemon=True).start()
        except ValueError:
            messagebox.showerror("Error", "Numeric fields must be integers.")

    run_btn = swiss_btn(left, "Run pipeline", save_and_run, primary=False, width=28)
    run_btn.config(bg=DARK, fg=WHITE, highlightbackground=DARK)
    run_btn.bind("<Enter>", lambda e: run_btn.config(bg="#333333", fg=WHITE, highlightbackground="#333333"))
    run_btn.bind("<Leave>", lambda e: run_btn.config(bg=DARK, fg=WHITE, highlightbackground=DARK))
    run_btn.pack(padx=20, pady=(4, 24))

    # ── RIGHT COLUMN ──────────────────────────────────────────────────────────
    right = tk.Frame(root, bg=WHITE)
    right.grid(row=1, column=1, sticky="nsew")
    right.grid_columnconfigure(0, weight=1)
    right.grid_rowconfigure(1, weight=2)   # chart
    right.grid_rowconfigure(4, weight=1)   # log

    # chart heading
    ch_head = tk.Frame(right, bg=WHITE)
    ch_head.grid(row=0, column=0, sticky="ew", padx=28, pady=(20, 0))
    tk.Label(ch_head, text="TIME SERIES PREVIEW",
             font=FONT_CAPS, bg=WHITE, fg=RED).pack(side="left")
    refresh_btn = tk.Button(ch_head, text="REFRESH",
                            command=lambda: update_plot(),
                            font=FONT_CAPS, bg=WHITE, fg=MID,
                            relief="flat", bd=0, cursor="hand2",
                            padx=6, pady=2,
                            highlightthickness=1, highlightbackground=RULE,
                            activebackground=RED, activeforeground=WHITE)
    refresh_btn.pack(side="right")
    tk.Frame(right, bg=DARK, height=1).grid(row=0, column=0, sticky="sew", padx=28, pady=(44, 0))

    # Matplotlib chart embedded in the right panel
    chart_frame = tk.Frame(right, bg=WHITE)
    chart_frame.grid(row=1, column=0, sticky="nsew", padx=28, pady=16)

    fig = Figure(figsize=(6.5, 3.0), dpi=100, facecolor=WHITE)
    ax  = fig.add_subplot(111)
    ax.set_facecolor(WHITE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    ax.tick_params(colors=MID, labelsize=8, length=3)
    ax.set_title("Load a dataset to preview",
                 color=MID, fontsize=8, fontfamily="Helvetica", loc="left", pad=6)
    fig.tight_layout(pad=1.8)
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_plot():
        """Reads the last 200 rows of the selected CSV and plots the target column."""
        if not dataset_path[0]:
            messagebox.showerror("Error", "Load a CSV dataset first.")
            return
        try:
            df     = pd.read_csv(dataset_path[0]).tail(200)
            target = entries["target_var"].get()
            if target not in df.columns:
                messagebox.showerror("Error", f"Column '{target}' not found.")
                return
            vals = df[target].values
            ax.clear()
            ax.set_facecolor(WHITE)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(RULE)
            ax.plot(vals, color=DARK, linewidth=1.0)
            ax.scatter(len(vals) - 1, vals[-1], color=RED, zorder=5, s=20)
            ax.tick_params(colors=MID, labelsize=8, length=3)
            ax.set_title(f"{target}  —  last {len(vals)} samples",
                         color=MID, fontsize=8, fontfamily="Helvetica", loc="left", pad=6)
            fig.tight_layout(pad=1.8)
            canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Plot failed:\n{e}")

    # log heading
    tk.Frame(right, bg=RULE, height=1).grid(row=2, column=0, sticky="ew", padx=28)
    log_head = tk.Frame(right, bg=WHITE)
    log_head.grid(row=3, column=0, sticky="ew", padx=28, pady=(10, 4))
    tk.Label(log_head, text="CONSOLE OUTPUT",
             font=FONT_CAPS, bg=WHITE, fg=RED).pack(side="left")

    # Console log widget — stdout and stderr are redirected here after this point
    log_frame = tk.Frame(right, bg=OFF)
    log_frame.grid(row=4, column=0, sticky="nsew", padx=28, pady=(0, 20))
    log_widget = tk.Text(log_frame, state="disabled",
                         font=FONT_MONO, bg=OFF, fg=DARK,
                         relief="flat", bd=0, padx=14, pady=10,
                         insertbackground=RED,
                         selectbackground=RED, selectforeground=WHITE)
    log_widget.pack(side="left", fill="both", expand=True)
    scrollbar = tk.Scrollbar(log_frame, orient="vertical",
                              command=log_widget.yview,
                              bg=OFF, troughcolor=OFF,
                              relief="flat", bd=0, width=6)
    scrollbar.pack(side="right", fill="y")
    log_widget.config(yscrollcommand=scrollbar.set)
    log_widget.tag_config("normal", foreground=DARK)
    log_widget.tag_config("accent", foreground=RED)
    log_widget.tag_config("error",  foreground=RED)

    _redir = TextRedirector(log_widget)
    sys.stdout = _redir
    sys.stderr = _redir          # catch Keras ANSI output on stderr too
    print("Ready. Configure parameters and select files to begin.\n")

    # ── FOOTER ────────────────────────────────────────────────────────────────
    tk.Frame(root, bg=DARK, height=1).grid(
        row=2, column=0, columnspan=2, sticky="ew")
    footer = tk.Frame(root, bg=WHITE)
    footer.grid(row=3, column=0, columnspan=2, sticky="ew", padx=28, pady=8)

    # Footer logo — silent fallback if icon is missing
    try:
        from PIL import Image, ImageTk

        def resource_path(relative):
            if hasattr(sys, "_MEIPASS"):
                return os.path.join(sys._MEIPASS, relative)
            return os.path.join(os.path.abspath("."), relative)

        logo_img = Image.open(resource_path("icona.icns")).resize((32, 32))
        logo_photo = ImageTk.PhotoImage(logo_img)
        logo_lbl = tk.Label(footer, image=logo_photo, bg=WHITE)
        logo_lbl.image = logo_photo
        logo_lbl.pack(side="left", padx=(0, 8))
    except Exception:
        pass

    tk.Label(footer, text="PREDATOR",
             font=FONT_CAPS, bg=WHITE, fg=DARK).pack(side="left")
    tk.Label(footer, text="DESIGNED BY ANTONIO MAZZITELLI",
             font=FONT_CAPS, bg=WHITE, fg=MID).pack(side="right")

    root.mainloop()


if __name__ == "__main__":
    create_gui()