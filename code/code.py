import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
#               НАСТРОЙКИ / РАЗМЕРНОСТЬ
# ============================================================

N_DEFAULT = 8  # по условию 8 компьютеров

# По заданию μ не определено, но для стационарного режима в цепи Маркова с непрерывным временем восстановление необходимо, поэтому указываем константу:
MU_REPAIR = 0.05


# ============================================================
#                   СЛУЖЕБНЫЕ СТРУКТУРЫ
# ============================================================

@dataclass
class ModelResult:
    states: List[Tuple[int, ...]]
    Q: np.ndarray
    P: np.ndarray
    Pk: np.ndarray
    mean_failed: float
    p_zero_failed: float


# ============================================================
# МАТЕМАТИЧЕСКАЯ МОДЕЛЬ (цепи Маркова с непрерывным временем: отказы + восстановления)
# Состояние: битовая строка длины N
# 0 - исправен, 1 - отказ
# Переходы:
#   0 -> 1 с интенсивностью λ_i
#   1 -> 0 с интенсивностью MU_REPAIR (общая константа)
# ============================================================

def generate_states(n: int) -> List[Tuple[int, ...]]:
    return [tuple(map(int, format(i, f"0{n}b"))) for i in range(2 ** n)]


def build_generator_matrix(
    states: List[Tuple[int, ...]],
    lambdas: List[float],
    mu_repair: float
) -> np.ndarray:
    n = len(lambdas)
    size = len(states)
    Q = np.zeros((size, size), dtype=float)
    index: Dict[Tuple[int, ...], int] = {s: i for i, s in enumerate(states)}

    for i, state in enumerate(states):
        rate_out = 0.0
        for k in range(n):
            new_state = list(state)

            if state[k] == 0:
                new_state[k] = 1
                j = index[tuple(new_state)]
                Q[i, j] += lambdas[k]
                rate_out += lambdas[k]
            else:
                new_state[k] = 0
                j = index[tuple(new_state)]
                Q[i, j] += mu_repair
                rate_out += mu_repair

        Q[i, i] = -rate_out

    return Q


def solve_stationary(Q: np.ndarray) -> np.ndarray:
    """
    Решаем πQ = 0, sum(π)=1
    Численно: (Q^T)*π = 0, последнюю строку заменяем на нормировку.
    """
    size = Q.shape[0]
    A = Q.T.copy()
    b = np.zeros(size, dtype=float)
    A[-1, :] = 1.0
    b[-1] = 1.0
    return np.linalg.solve(A, b)


def aggregate_by_failed(states: List[Tuple[int, ...]], P: np.ndarray, n: int) -> np.ndarray:
    Pk = np.zeros(n + 1, dtype=float)
    for s, p in zip(states, P):
        Pk[sum(s)] += p
    return Pk


def run_model(n: int, lambdas: List[float], mu_repair: float) -> ModelResult:
    states = generate_states(n)
    Q = build_generator_matrix(states, lambdas, mu_repair)
    P = solve_stationary(Q)
    Pk = aggregate_by_failed(states, P, n)
    mean_failed = float(sum(k * Pk[k] for k in range(n + 1)))
    p_zero_failed = float(Pk[0])

    return ModelResult(
        states=states,
        Q=Q,
        P=P,
        Pk=Pk,
        mean_failed=mean_failed,
        p_zero_failed=p_zero_failed
    )


# ============================================================
#               ГРАФ: ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def _layered_layout_by_k(states, x_step: float = 3.6, y_step: float = 1.45):
    """
    Размещение слоями по k = числу отказов:
      x = k * x_step
      y = (индекс внутри слоя) * y_step
    """
    by_k: Dict[int, List[Tuple[int, ...]]] = {}
    for s in states:
        by_k.setdefault(sum(s), []).append(s)

    for k in by_k:
        by_k[k].sort()

    pos = {}
    for k, layer in by_k.items():
        m = len(layer)
        for j, s in enumerate(layer):
            x = float(k) * x_step
            y = (j - (m - 1) / 2) * y_step
            pos[s] = (x, y)
    return pos


def state_to_int(bits: Tuple[int, ...]) -> int:
    return int("".join(map(str, bits)), 2)


def failed_set_label(bits: Tuple[int, ...]) -> str:
    failed = [str(i + 1) for i, b in enumerate(bits) if b == 1]
    return ",".join(failed) if failed else ""


def draw_full_state_graph_page(
    n: int,
    lambdas: List[float],
    mu_repair: float,
    only_i: Optional[int],
    ax: plt.Axes
) -> None:
    """
    Рисует граф на ax.
    only_i:
      - None: рисуем дуги для всех i.
      - 1..n: рисуем дуги/подписи только для одного i.
    """
    states = generate_states(n)
    G = nx.DiGraph()
    for s in states:
        G.add_node(s, k=sum(s))

    # Размещение
    pos = _layered_layout_by_k(states, x_step=3.8, y_step=1.9)

    fail_edges, repair_edges = [], []
    fail_labels, repair_labels = {}, {}

    for s in states:
        for i in range(n):
            if only_i is not None and (i + 1) != only_i:
                continue

            t = list(s)
            if s[i] == 0:
                t[i] = 1
                u, v = s, tuple(t)
                fail_edges.append((u, v))
                fail_labels[(u, v)] = rf"$\lambda_{i+1}$"
                G.add_edge(u, v)
            else:
                t[i] = 0
                u, v = s, tuple(t)
                repair_edges.append((u, v))
                repair_labels[(u, v)] = rf"$\mu_{i+1}$"
                G.add_edge(u, v)

    # Узлы
    nx.draw_networkx_nodes(
        G, pos,
        ax=ax,
        node_size=700,
        node_color="#f1c40f",
        edgecolors="black",
        linewidths=1.0
    )

    # Дуги: красные и зелёные
    alpha = 0.55 if only_i is None else 0.85
    rad = 0.18 if only_i is None else 0.22

    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        edgelist=fail_edges,
        width=0.55,
        edge_color="crimson",
        alpha=alpha,
        arrows=True,
        arrowsize=8,
        connectionstyle=f"arc3,rad={rad}"
    )
    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        edgelist=repair_edges,
        width=0.55,
        edge_color="seagreen",
        alpha=alpha,
        arrows=True,
        arrowsize=8,
        connectionstyle=f"arc3,rad={-rad}"
    )

    # Подписи узлов: номер состояния
    node_numbers = {s: str(state_to_int(s)) for s in states}
    nx.draw_networkx_labels(
        G, pos,
        ax=ax,
        labels=node_numbers,
        font_size=5,
        font_color="black"
    )

    # Над узлом: множество отказавших
    for s in states:
        top = failed_set_label(s)
        if top:
            x, y = pos[s]
            ax.text(
                x, y + 0.50,
                top,
                ha="center", va="bottom",
                fontsize=4,
                color="black"
            )

    # Подписи дуг
    fs = 7 if only_i is not None else 3
    nx.draw_networkx_edge_labels(
        G, pos,
        ax=ax,
        edge_labels=fail_labels,
        font_size=fs,
        rotate=False,
        label_pos=0.55,
        font_color="crimson",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.45, pad=0.10)
    )
    nx.draw_networkx_edge_labels(
        G, pos,
        ax=ax,
        edge_labels=repair_labels,
        font_size=fs,
        rotate=False,
        label_pos=0.45,
        font_color="seagreen",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.45, pad=0.10)
    )

    suffix = f"(i={only_i})" if only_i is not None else "(все i)"
    ax.set_title(
        f"Граф состояний {suffix}, N={n}, |S|={2**n}\n"
        f"Красные: λ_i, зелёные: μ_i (в модели μ={mu_repair})"
    )
    ax.set_axis_off()


def save_full_graph_pdf_per_i(n: int, lambdas: List[float], mu_repair: float, pdf_path: str) -> None:
    """
    Сохраняет PDF, где 8 страниц (i=1..n).
    На каждой странице:
      - все узлы (256 состояний)
      - только дуги/подписи для конкретного i
    """
    with PdfPages(pdf_path) as pdf:
        for i in range(1, n + 1):
            fig, ax = plt.subplots(figsize=(30, 18))
            draw_full_state_graph_page(n, lambdas, mu_repair, only_i=i, ax=ax)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


# ============================================================
#       GUI: ПРОКРУЧИВАЕМЫЙ ФРЕЙМ ДЛЯ ВВОДА λ1..λN
# ============================================================

class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, height: int = 220):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, height=height, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)

        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", self._on_configure)

        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.v_scroll.pack(side="right", fill="y")

        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

    def _on_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


# ============================================================
#               GUI: ОСНОВНОЕ ПРИЛОЖЕНИЕ
# ============================================================

class SafetySystemApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.n = N_DEFAULT
        self.last_result: Optional[ModelResult] = None

        self.title(f"Теория информационных процессов и систем — GUI (N = {self.n})")
        self.geometry("1100x720")

        self._create_menu()
        self._create_widgets()
        self._fill_defaults()

    # --------------------------
    #           MENU
    # --------------------------

    def _create_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Загрузить λ из файла...", command=self.load_lambdas)
        file_menu.add_command(label="Сохранить λ в файл...", command=self.save_lambdas)
        file_menu.add_separator()
        file_menu.add_command(label="Экспорт результатов (CSV)...", command=self.export_results_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.destroy)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Граф: сохранить PDF (8 страниц i=1..8)", command=self.graph_pdf)
        view_menu.add_command(label="Граф: показать общий", command=self.graph_all)
        view_menu.add_command(label="Граф: сохранить общий в PNG", command=self.graph_png)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Размерность (справка)", command=self.show_dimension_info)
        help_menu.add_command(label="О программе", command=self.show_about)

        menubar.add_cascade(label="Файл", menu=file_menu)
        menubar.add_cascade(label="Вид", menu=view_menu)
        menubar.add_cascade(label="Справка", menu=help_menu)

        self.config(menu=menubar)

    def show_dimension_info(self):
        num_states = 2 ** self.n
        msg = (
            "Справочная информация (размерность модели)\n\n"
            f"Количество компьютеров N = {self.n}\n"
            f"Число состояний = 2^N = {num_states}\n"
            f"Размер матрицы генератора Q: {num_states} x {num_states}\n\n"
            "Состояние кодируется битами длины N:\n"
            "0 — исправен, 1 — отказ.\n\n"
            "Восстановление используется как константа:\n"
            f"MU_REPAIR = {MU_REPAIR}\n"
            "(μ не вводится в интерфейсе)."
        )
        messagebox.showinfo("Размерность", msg)

    def show_about(self):
        messagebox.showinfo(
            "О программе",
            "- ввод λ1..λN\n"
            "- прокручиваемые панели\n"
            "- меню со справочной информацией о размерности\n"
            "- результаты в этом же окне\n"
            "- граф состояний + сохранение PDF на 8 страниц (по i)"
        )

    # --------------------------
    #           LAYOUT
    # --------------------------

    def _create_widgets(self):
        params_frame = ttk.LabelFrame(self, text="Интенсивности отказов (вводятся только λ1..λN)")
        params_frame.pack(fill="x", padx=10, pady=10)

        self.sf = ScrollableFrame(params_frame, height=210)
        self.sf.pack(fill="x", padx=8, pady=8)

        self.lambda_entries: List[ttk.Entry] = []

        for i in range(self.n):
            ttk.Label(self.sf.inner, text=f"λ{i+1}").grid(row=i, column=0, sticky="w", padx=6, pady=4)
            e = ttk.Entry(self.sf.inner, width=12)
            e.grid(row=i, column=1, sticky="w", padx=6, pady=4)
            self.lambda_entries.append(e)
            ttk.Label(self.sf.inner, text="(1/час)").grid(row=i, column=2, sticky="w", padx=6, pady=4)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Button(btn_frame, text="Рассчитать", command=self.calculate).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="PDF граф (8 стр.)", command=self.graph_pdf).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Показать общий граф", command=self.graph_all).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Сохранить общий граф (PNG)", command=self.graph_png).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Копировать результаты", command=self.copy_results).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Очистить результаты", command=self.clear_results).pack(side="left", padx=5)

        results_frame = ttk.LabelFrame(self, text="Результаты (в этом же окне)")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)

        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=2)
        results_frame.rowconfigure(0, weight=1)

        # Слева: P(k)
        left_frame = ttk.Frame(results_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)

        ttk.Label(left_frame, text="Агрегированные вероятности P(k), k=0..N").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )

        self.pk_tree = ttk.Treeview(left_frame, columns=("k", "Pk"), show="headings", height=10)
        self.pk_tree.heading("k", text="k (отказов)")
        self.pk_tree.heading("Pk", text="P(k)")
        self.pk_tree.column("k", width=90, anchor="center")
        self.pk_tree.column("Pk", width=180, anchor="w")

        pk_scroll = ttk.Scrollbar(left_frame, orient="vertical", command=self.pk_tree.yview)
        self.pk_tree.configure(yscrollcommand=pk_scroll.set)

        self.pk_tree.grid(row=1, column=0, sticky="nsew")
        pk_scroll.grid(row=1, column=1, sticky="ns")

        # Справа: полный вывод
        right_frame = ttk.Frame(results_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)

        ttk.Label(right_frame, text="Полные результаты (состояния и стационарные вероятности)").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )

        self.text = tk.Text(right_frame, wrap="none")
        text_scroll_y = ttk.Scrollbar(right_frame, orient="vertical", command=self.text.yview)
        text_scroll_x = ttk.Scrollbar(right_frame, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=text_scroll_y.set, xscrollcommand=text_scroll_x.set)

        self.text.grid(row=1, column=0, sticky="nsew")
        text_scroll_y.grid(row=1, column=1, sticky="ns")
        text_scroll_x.grid(row=2, column=0, sticky="ew")

        self.status = ttk.Label(self, text="Готово", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(0, 8))

    def _fill_defaults(self):
        for e in self.lambda_entries:
            e.delete(0, tk.END)
            e.insert(0, "0.01")

    # --------------------------
    #     ВВОД / ВАЛИДАЦИЯ
    # --------------------------

    def get_lambdas(self) -> List[float]:
        lambdas: List[float] = []
        for i, e in enumerate(self.lambda_entries, start=1):
            raw = e.get().strip().replace(",", ".")
            if not raw:
                raise ValueError(f"λ{i} пусто")
            val = float(raw)
            if val < 0:
                raise ValueError(f"λ{i} < 0 недопустимо")
            lambdas.append(val)
        return lambdas

    # --------------------------
    #          ACTIONS
    # --------------------------

    def calculate(self):
        try:
            lambdas = self.get_lambdas()
        except Exception as ex:
            messagebox.showerror("Ошибка ввода", f"Некорректные λ:\n{ex}")
            return

        try:
            self.status.config(text="Расчёт... (256 состояний)")
            self.update_idletasks()

            result = run_model(self.n, lambdas, MU_REPAIR)
            self.last_result = result

            self._render_results(lambdas, result)
            self.status.config(text="Расчёт выполнен")
        except np.linalg.LinAlgError as ex:
            self.status.config(text="Ошибка расчёта")
            messagebox.showerror("Ошибка", f"Не удалось решить систему:\n{ex}")
        except Exception as ex:
            self.status.config(text="Ошибка расчёта")
            messagebox.showerror("Ошибка", f"Неожиданная ошибка:\n{ex}")

    def _render_results(self, lambdas: List[float], result: ModelResult):
        for item in self.pk_tree.get_children():
            self.pk_tree.delete(item)

        for k in range(self.n + 1):
            self.pk_tree.insert("", "end", values=(k, f"{result.Pk[k]:.10e}"))

        self.text.delete("1.0", tk.END)

        num_states = 2 ** self.n
        self.text.insert(tk.END, "=== Справка / Размерность ===\n")
        self.text.insert(tk.END, f"N = {self.n}\n")
        self.text.insert(tk.END, f"Число состояний: 2^N = {num_states}\n")
        self.text.insert(tk.END, f"MU_REPAIR (общая, не вводится в GUI): {MU_REPAIR}\n\n")

        self.text.insert(tk.END, "=== Введённые интенсивности отказов ===\n")
        for i, lam in enumerate(lambdas, start=1):
            self.text.insert(tk.END, f"λ{i} = {lam}\n")

        self.text.insert(tk.END, "\n=== Интегральные показатели ===\n")
        self.text.insert(tk.END, f"P(0 отказов) = {result.p_zero_failed:.10e}\n")
        self.text.insert(tk.END, f"Мат. ожидание числа отказавших = {result.mean_failed:.6f}\n")
        self.text.insert(tk.END, f"Сумма вероятностей (контроль) = {result.P.sum():.12f}\n\n")

        self.text.insert(tk.END, "=== Стационарные вероятности всех состояний ===\n")
        self.text.insert(tk.END, "(Формат: состояние (0/1) : P)\n\n")

        for s, p in zip(result.states, result.P):
            self.text.insert(tk.END, f"{s} : P = {p:.12e}\n")

    def graph_pdf(self):
        """Сохранить PDF на 8 страниц (по i=1..8)."""
        try:
            lambdas = self.get_lambdas()
        except Exception as ex:
            messagebox.showerror("Ошибка ввода", f"Некорректные λ:\n{ex}")
            return

        pdf_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("All files", "*.*")]
        )
        if not pdf_path:
            return

        try:
            self.status.config(text="Формирование PDF (8 страниц)...")
            self.update_idletasks()

            save_full_graph_pdf_per_i(self.n, lambdas, MU_REPAIR, pdf_path)

            self.status.config(text="PDF сохранён")
            messagebox.showinfo("Готово", f"PDF сохранён:\n{pdf_path}")
        except Exception as ex:
            self.status.config(text="Ошибка построения PDF")
            messagebox.showerror("Ошибка", f"Не удалось построить/сохранить PDF:\n{ex}")
            
    def graph_png(self):
        """Сохранить общий граф (все i) в PNG."""
        try:
            lambdas = self.get_lambdas()
        except Exception as ex:
            messagebox.showerror("Ошибка ввода", f"Некорректные λ:\n{ex}")
            return

        png_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")]
        )
        if not png_path:
            return

        try:
            self.status.config(text="Сохранение PNG (общий граф)...")
            self.update_idletasks()

            fig, ax = plt.subplots(figsize=(30, 18))
            draw_full_state_graph_page(self.n, lambdas, MU_REPAIR, only_i=None, ax=ax)
            fig.tight_layout()
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            self.status.config(text="PNG сохранён")
            messagebox.showinfo("Готово", f"PNG сохранён:\n{png_path}")
        except Exception as ex:
            self.status.config(text="Ошибка сохранения PNG")
            messagebox.showerror("Ошибка", f"Не удалось сохранить PNG:\n{ex}")

    def graph_all(self):
        """Показать общий граф (все i) — будет слипание при N=8 (это нормально)."""
        try:
            lambdas = self.get_lambdas()
        except Exception as ex:
            messagebox.showerror("Ошибка ввода", f"Некорректные λ:\n{ex}")
            return

        fig, ax = plt.subplots(figsize=(30, 18))
        draw_full_state_graph_page(self.n, lambdas, MU_REPAIR, only_i=None, ax=ax)
        fig.tight_layout()
        plt.show()

    def copy_results(self):
        data = self.text.get("1.0", tk.END).strip()
        if not data:
            messagebox.showwarning("Нет данных", "Сначала выполните расчёт.")
            return
        self.clipboard_clear()
        self.clipboard_append(data)
        self.update()
        messagebox.showinfo("Готово", "Результаты скопированы в буфер обмена.")

    def clear_results(self):
        self.text.delete("1.0", tk.END)
        for item in self.pk_tree.get_children():
            self.pk_tree.delete(item)
        self.last_result = None
        self.status.config(text="Результаты очищены")

    # --------------------------
    #   ОПЕРАЦИИ С ФАЙЛАМИ
    # --------------------------

    def save_lambdas(self):
        try:
            lambdas = self.get_lambdas()
        except Exception as ex:
            messagebox.showerror("Ошибка", f"Некорректные λ:\n{ex}")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Text", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(["i", "lambda"])
                for i, lam in enumerate(lambdas, start=1):
                    w.writerow([i, lam])
            messagebox.showinfo("Сохранено", f"λ сохранены в файл:\n{path}")
        except Exception as ex:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{ex}")

    def load_lambdas(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv"), ("Text", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            values: List[float] = []

            if path.lower().endswith(".csv"):
                with open(path, "r", encoding="utf-8") as f:
                    r = csv.reader(f, delimiter=";")
                    rows = list(r)

                for row in rows:
                    if not row:
                        continue
                    if row[0].strip().lower() in ("i", "index"):
                        continue
                    if len(row) == 1:
                        values.append(float(row[0].strip().replace(",", ".")))
                    else:
                        values.append(float(row[1].strip().replace(",", ".")))
            else:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        values.append(float(line.replace(",", ".")))

            if len(values) < self.n:
                raise ValueError(f"В файле {len(values)} значений, а нужно минимум {self.n}.")

            values = values[:self.n]
            for e, v in zip(self.lambda_entries, values):
                e.delete(0, tk.END)
                e.insert(0, str(v))

            messagebox.showinfo("Загружено", f"λ загружены из файла:\n{path}")

        except Exception as ex:
            messagebox.showerror("Ошибка", f"Не удалось загрузить λ:\n{ex}")

    def export_results_csv(self):
        if not self.last_result:
            messagebox.showwarning("Нет данных", "Сначала выполните расчёт.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            r = self.last_result
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter=";")

                w.writerow(["N", self.n])
                w.writerow(["num_states", 2 ** self.n])
                w.writerow(["MU_REPAIR", MU_REPAIR])
                w.writerow([])

                w.writerow(["k", "P(k)"])
                for k in range(self.n + 1):
                    w.writerow([k, r.Pk[k]])

                w.writerow([])
                w.writerow(["state_bits", "P(state)"])
                for s, p in zip(r.states, r.P):
                    w.writerow(["".join(map(str, s)), p])

            messagebox.showinfo("Экспорт", f"Результаты экспортированы:\n{path}")

        except Exception as ex:
            messagebox.showerror("Ошибка", f"Не удалось экспортировать:\n{ex}")


# ==============================
#           ЗАПУСК
# ==============================

if __name__ == "__main__":
    app = SafetySystemApp()
    app.mainloop()
