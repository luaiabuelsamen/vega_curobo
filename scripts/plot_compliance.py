"""Plot the compliance A/B from media/compliance.csv.

    python scripts/plot_compliance.py     # -> media/compliance_force.png

Two panels rather than two y-axes on one: contact force and joint torque are
different quantities, and overlaying them on a shared frame would invent a
comparison that does not exist.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
STIFF = "#eb6834"
COMPLIANT = "#2a78d6"


def smooth(t, y, window):
    """Rolling mean, with the edges dropped rather than tapered -- a partial
    window at the tail dives toward zero and reads as a real drop in force."""
    kernel = np.ones(window) / window
    edge = window // 2
    return t[edge:-edge], np.convolve(y, kernel, mode="same")[edge:-edge]


def main(source="media/compliance.csv", out="media/compliance_force.png"):
    data = np.genfromtxt(source, delimiter=",", names=True)
    t = data["t"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.0), sharex=True,
                             gridspec_kw={"hspace": 0.30, "top": 0.84})
    fig.patch.set_facecolor(SURFACE)

    panels = [
        (axes[0], "stiff_force", "compliant_force", "Contact force on the wall", "N"),
        (axes[1], "stiff_torque", "compliant_torque", "Peak joint torque", "N·m"),
    ]

    contact_start = t[np.argmax(data["stiff_force"] > 1.0)]

    for ax, stiff_key, compliant_key, title, unit in panels:
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)

        ax.axvline(contact_start, color=GRID, linewidth=1.4, zorder=0)
        # Rigid contact chatters at every step. Raw underneath so nothing is
        # hidden, a 50 ms rolling mean on top so the level is readable.
        for key, colour in ((stiff_key, STIFF), (compliant_key, COMPLIANT)):
            ax.plot(t, data[key], color=colour, linewidth=0.7, alpha=0.25)
            ax.plot(*smooth(t, data[key], 25), color=colour, linewidth=2.0)

        ax.set_title(f"{title}  ({unit})", color=INK, fontsize=11,
                     loc="left", pad=8, fontweight="medium")

        # direct labels beat hunting through a legend
        for key, colour, name in ((stiff_key, STIFF, "stiff"),
                                  (compliant_key, COMPLIANT, "compliant")):
            tail = data[key][-1]
            ax.annotate(name, xy=(t[-1], tail), xytext=(6, 0),
                        textcoords="offset points", color=colour,
                        fontsize=9, va="center", fontweight="medium")
        ax.set_xlim(t[0], t[-1] * 1.12)

    axes[0].annotate("contact", xy=(contact_start, axes[0].get_ylim()[1] * 0.92),
                     xytext=(6, 0), textcoords="offset points",
                     color=INK_SOFT, fontsize=9)
    axes[1].set_xlabel("time (s)", color=INK_SOFT, fontsize=9)

    fig.suptitle("Same plan, same wall, two controllers",
                 color=INK, fontsize=13, x=0.125, ha="left", y=0.99,
                 fontweight="semibold")
    fig.text(0.125, 0.945,
             "Impedance control does not soften the impact — it stops the arm leaning on the obstacle.",
             color=INK_SOFT, fontsize=9.5, ha="left")

    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
