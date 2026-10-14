"""Glossary: short explanations with typeset formulas, opened from links in the text.

In any label, write [[term|shown text]] (or [[term]]) to link to an entry.
Entries are lists of blocks: ("p", html paragraph) or ("eq", LaTeX formula
rendered with matplotlib's mathtext, so no TeX installation is needed).
"""
from __future__ import annotations

import io
import re
from functools import lru_cache

ENTRIES: dict[str, tuple[str, list]] = {
    "notation": ("Notation used throughout", [
        ("p", "<b>Configurations.</b> σ = (σ<sub>1</sub>, …, σ<sub>N</sub>), each spin σ<sub>i</sub> = ±1. "
              "Energy (Hamiltonian) H(σ), written H<sub>N</sub> when the size N matters. "
              "β = 1/T is the inverse temperature."),
        ("p", "<b>Measures.</b> μ: the reference (a priori) distribution, e.g. uniform on {−1, +1}<sup>N</sup> or a standard "
              "Gaussian. μ<sub>β</sub>: the Gibbs measure. ν: a trial distribution; in robust optimisation, the adversary's choice."),
        ("eq", r"$\mu_\beta(\sigma)=\frac{e^{-\beta H(\sigma)}}{Z},\qquad Z=\sum_\sigma e^{-\beta H(\sigma)}$"),
        ("p", "<b>Information.</b> Ent(ν): entropy. H(ν | μ): relative entropy, also called the Kullback–Leibler (KL) "
              "divergence. The letter H with a bar and two measures is always relative entropy, never the energy. "
              "ρ: the entropy budget, ρ = log 2 for REM-type models."),
        ("eq", r"$\mathrm{Ent}(\nu)=-\sum_\sigma \nu(\sigma)\log\nu(\sigma),\qquad H(\nu\,|\,\mu)=\sum_\sigma \nu(\sigma)\log\frac{\nu(\sigma)}{\mu(\sigma)}$"),
        ("p", "<b>Order parameters.</b> m = (1/N) Σ σ<sub>i</sub>: magnetisation (Curie–Weiss). "
              "m<sub>μ</sub> = (1/N) Σ ξ<sub>i</sub><sup>μ</sup> σ<sub>i</sub>: overlap with the stored pattern ξ<sup>μ</sup>, "
              "μ = 1, …, P, load α = P/N (Hopfield). q: overlap between two replicas. In the Parisi-type formula the "
              "pair (q, m) is the dual variable: q ∈ [0, 1] an overlap, m ∈ (0, 1] the replica-symmetry-breaking parameter "
              "(the m = β<sub>c</sub>/β of the REM). Context tells the two m's apart."),
    ]),
    "spin": ("Spin", [
        ("p", "A variable that takes two values, σ = +1 or σ = −1: up or down, on or off, yes or no. "
              "A configuration of N spins is a vector σ ∈ {−1, +1}<sup>N</sup>, so there are 2<sup>N</sup> configurations."),
        ("eq", r"$\sigma=(\sigma_1,\dots,\sigma_N)\in\{-1,+1\}^N$"),
    ]),
    "energy": ("Energy (Hamiltonian)", [
        ("p", "A function that scores each configuration: low energy means preferred. Physics calls it the "
              "Hamiltonian; machine learning would call it a (negative) score or a loss."),
        ("eq", r"$H_N:\{-1,+1\}^N\to\mathbb{R},\qquad \sigma\mapsto H_N(\sigma)$"),
        ("p", "Energy alone does not fix the probabilities: the inverse temperature β = 1/T says how much it counts. "
              "The Gibbs distribution turns energies into probabilities, lower energy meaning more likely."),
        ("eq", r"$\mu_\beta(\sigma)=\frac{e^{-\beta H(\sigma)}}{Z},\qquad \log Z=-\beta\,\mathbb{E}_{\mu_\beta}[H]+\mathrm{Ent}(\mu_\beta)$"),
        ("demo", "energy"),
    ]),
    "temperature": ("Temperature and β", [
        ("p", "The temperature T decides how strictly the system follows the energy; β = 1/T is the inverse "
              "temperature. At β = 0 every configuration is equally likely; as β → ∞ only the lowest-energy "
              "configurations survive."),
    ]),
    "gibbs": ("Gibbs measure", [
        ("p", "The probability of a configuration at inverse temperature β. It is the softmax of −βH over all "
              "configurations; the normaliser Z is the partition function."),
        ("eq", r"$\mu_{\beta}(\sigma)=\frac{e^{-\beta H_N(\sigma)}}{Z_N(\beta)},\qquad Z_N(\beta)=\sum_{\sigma} e^{-\beta H_N(\sigma)}$"),
    ]),
    "free-energy": ("Free energy", [
        ("p", "The logarithm of the partition function, per spin. It is the central quantity: its derivatives give "
              "the average energy, the magnetisation and every other observable, and its non-smooth points are the "
              "phase transitions. (Physicists put a factor −1/β in front; here we use the probabilists' sign.)"),
        ("eq", r"$f_N(\beta)=\frac{1}{N}\log Z_N(\beta)$"),
        ("p", "Gibbs variational principle: the free energy is the best trade-off between energy and entropy, "
              "over all trial distributions ν; the maximiser is the Gibbs measure."),
        ("eq", r"$\log Z_N(\beta)=\max_{\nu}\left\{-\beta\,\mathbb{E}_\nu[H_N]+\mathrm{Ent}(\nu)\right\}$"),
    ]),
    "entropy": ("Entropy", [
        ("p", "How spread out a probability distribution is. It is largest (log n) for the uniform distribution "
              "on n states and zero for a distribution concentrated on one state. With μ uniform, "
              "Ent(ν) = log n − H(ν | μ): entropy is relative entropy seen from the uniform distribution."),
        ("eq", r"$\mathrm{Ent}(\nu)=-\sum_{\sigma} \nu(\sigma)\log \nu(\sigma)$"),
        ("p", "Why: with μ = 1/n, log(ν/μ) = log ν + log n, so H(ν | μ) = Σ ν log ν + log n = log n − Ent(ν). "
              "Entropy and distance from uniform share a fixed total, log n."),
        ("eq", r"$\mathrm{Ent}(\nu)+H(\nu\,|\,\mu)=\log n\qquad(\mu\ \mathrm{uniform\ on\ }n\ \mathrm{states})$"),
        ("demo", "entropy"),
    ]),
    "kl": ("Relative entropy (Kullback–Leibler divergence)", [
        ("p", "How far a distribution ν is from a reference μ. It is never negative and is zero only when ν = μ. "
              "In robust optimisation a ball {H(ν | μ) ≤ ρ} is the set of distributions an adversary may choose."),
        ("eq", r"$H(\nu\,|\,\mu)=\int \log\frac{d\nu}{d\mu}\,d\nu\;\geq\;0$"),
        ("p", "Sanov's theorem: the probability that N independent samples from μ look like ν decays as "
              "e<sup>−N H(ν | μ)</sup>."),
    ]),
    "order-parameter": ("Order parameter", [
        ("p", "A few numbers that summarise the macroscopic state and change behaviour at a phase transition: "
              "the magnetisation in a magnet, the overlaps with the stored memories in a Hopfield network, the "
              "overlap between two replicas in a spin glass."),
    ]),
    "magnetisation": ("Magnetisation", [
        ("p", "The average spin. m ≈ 0: no collective preference; m close to ±1: almost everyone agrees."),
        ("eq", r"$m_N(\sigma)=\frac{1}{N}\sum_{i=1}^N \sigma_i\in[-1,1]$"),
    ]),
    "overlap": ("Overlap", [
        ("p", "The similarity of two configurations, between −1 (opposite) and +1 (identical). The overlap with a "
              "stored pattern ξ measures recall in a Hopfield network; the overlap between two independent copies "
              "(replicas) measures how the Gibbs measure is organised in a spin glass."),
        ("eq", r"$q_N(\sigma,\tau)=\frac{1}{N}\sum_{i=1}^N\sigma_i\tau_i,\qquad m_\mu(\sigma)=\frac{1}{N}\sum_{i=1}^N\xi_i^\mu\sigma_i$"),
    ]),
    "phase-transition": ("Phase transition", [
        ("p", "A value of a parameter (here usually β) where the large-N behaviour changes abruptly: a magnet "
              "starts to magnetise, a memory becomes retrievable, a glass freezes. Mathematically the limiting free "
              "energy is not analytic there."),
    ]),
    "curie-weiss": ("Curie–Weiss model", [
        ("p", "The simplest model of collective order: every spin interacts equally with every other spin. "
              "Its free energy is an explicit one-dimensional variational problem over the magnetisation m."),
        ("eq", r"$H_N(\sigma)=-\frac{N}{2}\,m_N(\sigma)^2-h\sum_i\sigma_i$"),
        ("eq", r"$\lim_{N\to\infty}\frac{1}{N}\log Z_N=\log 2+\sup_{m\in[-1,1]}\left\{\frac{\beta m^2}{2}+\beta h m-I(m)\right\}$"),
        ("eq", r"$I(m)=\frac{1+m}{2}\log(1+m)+\frac{1-m}{2}\log(1-m)$"),
        ("p", "I(m) is the large-deviation rate of the mean of fair ±1 coins (Cramér). The maximisers solve "
              "m = tanh(β(m + h)); for h = 0 there is a phase transition at β = 1."),
    ]),
    "hopfield": ("Hopfield model", [
        ("p", "An associative memory: P patterns ξ<sup>1</sup>, …, ξ<sup>P</sup> ∈ {−1, +1}<sup>N</sup> are stored in the couplings by Hebb's rule, "
              "and the energy is low near each stored pattern. With P = 1 it is the Curie–Weiss model in disguise "
              "(flip σ<sub>i</sub> ↦ ξ<sub>i</sub>σ<sub>i</sub>); with P proportional to N it becomes glassy."),
        ("eq", r"$H_N(\sigma)=-\frac{N}{2}\sum_{\mu=1}^{P} m_\mu(\sigma)^2,\qquad m_\mu(\sigma)=\frac{1}{N}\sum_i\xi_i^\mu\sigma_i$"),
        ("p", "Low load (P fixed): the free energy is a supremum over the overlaps, a P-dimensional Curie–Weiss formula."),
        ("eq", r"$\lim\frac{1}{N}\log Z_N=\sup_{m\in\mathbb{R}^P}\left\{-\frac{\beta|m|^2}{2}+\mathbb{E}_\xi\log 2\cosh(\beta\, m\cdot\xi)\right\}$"),
    ]),
    "hebb": ("Hebb's rule", [
        ("p", "'Neurons that fire together wire together': the coupling between two neurons is the correlation of "
              "their states across the stored patterns."),
        ("eq", r"$J_{ij}=\frac{1}{N}\sum_{\mu=1}^{P}\xi_i^\mu\xi_j^\mu$"),
    ]),
    "rbm": ("Restricted Boltzmann machine", [
        ("p", "A two-layer probabilistic model with visible units σ and hidden units z and links only between the "
              "layers ('restricted': no links inside a layer). With Gaussian hidden units its visible marginal is "
              "exactly a Hopfield model (Barra, Bernacchia, Santucci, Contucci 2012)."),
        ("eq", r"$P(\sigma,z)\propto \exp\!\left(-\frac{1}{2}\sum_\mu z_\mu^2+\sqrt{\frac{\beta}{N}}\sum_{i,\mu}\xi_i^\mu \sigma_i z_\mu\right)$"),
        ("eq", r"$\int e^{-\frac{z^2}{2}+a z}\,\frac{dz}{\sqrt{2\pi}}=e^{a^2/2}$"),
        ("p", "This Gaussian identity is a Legendre transform in disguise: the hidden unit is the dual variable of "
              "the convex function m ↦ m²/2."),
    ]),
    "legendre": ("Legendre transform and duality", [
        ("p", "A convex function is the maximum of the lines below it, a concave one the minimum of the lines "
              "above it. Writing a non-linearity this way introduces a dual variable. Convex (Curie–Weiss, "
              "Hopfield) gives a supremum; concave (the REM approximation of TAP) gives an infimum, the "
              "Parisi-type formula."),
        ("eq", r"$\frac{m^2}{2}=\sup_{z}\left\{z\,m-\frac{z^2}{2}\right\},\qquad \Phi(x)=\inf_{q}\left\{\Phi(q)+\Phi'(q)(x-q)\right\}\ \ (\Phi\ \mathrm{concave})$"),
    ]),
    "capacity": ("Storage capacity", [
        ("p", "How many patterns per neuron a Hopfield network can store and still recall. With random patterns "
              "and P = αN, recall survives up to α<sub>c</sub> ≈ 0.138 at zero temperature (Amit, Gutfreund, "
              "Sompolinsky, replica-symmetric). Replica-symmetry breaking moves it only slightly "
              "(Steffan and Kühn: ≈ 0.13819). A rigorous derivation of α<sub>c</sub> is still open."),
    ]),
    "spin-glass": ("Spin glass", [
        ("p", "A magnetic system with random couplings, some favouring agreement and others disagreement, that "
              "freezes at low temperature into a disordered state. Random couplings alone do not make a glass: "
              "the key ingredients are disorder and frustration (no configuration satisfies all couplings), and "
              "the glassy behaviour appears only below a critical temperature. In the mean-field theory of "
              "Parisi, the Gibbs measure then splits into many 'pure states' organised in a tree "
              "(ultrametricity)."),
    ]),
    "sk": ("Sherrington–Kirkpatrick (SK) model", [
        ("p", "The mean-field spin glass: Gaussian couplings between all pairs. Its free energy is given by the "
              "Parisi formula (proved by Guerra and Talagrand). A heavily loaded Hopfield network behaves like it: "
              "the cross-talk between many random patterns acts as random couplings."),
        ("eq", r"$H_N(\sigma)=-\frac{1}{\sqrt{N}}\sum_{i<j}g_{ij}\sigma_i\sigma_j,\qquad g_{ij}\ \mathrm{i.i.d.}\ \mathcal{N}(0,1)$"),
        ("eq", r"$\lim_{N\to\infty}\frac{1}{N}\log Z_N=\inf_{x}\ \mathcal{P}_\beta(x)$"),
    ]),
    "rem": ("Random Energy Model (REM)", [
        ("p", "Derrida's model of extreme disorder: every configuration gets its own independent Gaussian energy. "
              "It freezes at β<sub>c</sub> = √(2 log 2), where the Gibbs measure concentrates on a handful of "
              "configurations."),
        ("eq", r"$H_N(\alpha)\ \mathrm{i.i.d.}\ \mathcal{N}(0,N),\quad \alpha=1,\dots,2^N$"),
        ("eq", r"$\lim\frac{1}{N}\log Z_N=\log 2+\frac{\beta^2}{2}\quad(\beta\leq\beta_c),\qquad =\beta\,\beta_c\quad(\beta>\beta_c)$"),
    ]),
    "glauber": ("Glauber dynamics", [
        ("p", "A Markov chain that samples the Gibbs measure: pick a spin at random and redraw it from its conditional "
              "law given all the others (a heat-bath update). Here h<sub>i</sub> is the local field felt by spin i. "
              "At T = 0 the spin simply aligns with its field: the energy never goes up."),
        ("eq", r"$P(\sigma_i=+1\mid \sigma_{j\neq i})=\frac{1}{1+e^{-2\beta h_i}},\qquad h_i=\sum_{j\neq i}J_{ij}\sigma_j+h$"),
    ]),
    "large-deviations": ("Large deviations (Cramér)", [
        ("p", "The probability that an average of N independent samples lands far from its mean decays exponentially "
              "in N; the exponent is the rate function. For fair ±1 coins (Cramér's theorem):"),
        ("eq", r"$P\left(\frac{1}{N}\sum_{i=1}^N\sigma_i\approx m\right)\approx e^{-N I(m)},\quad I(m)=\frac{1+m}{2}\log(1+m)+\frac{1-m}{2}\log(1-m)$"),
        ("p", "Combined with the Laplace principle (on the log scale, a sum of terms e<sup>N a<sub>k</sub></sup> is its "
              "largest term), this turns free energies into variational formulas. Sanov's theorem is the version for "
              "whole empirical distributions, with rate H(ν | μ)."),
        ("eq", r"$\frac{1}{N}\log\sum_k e^{N a_k}\ \to\ \max_k a_k$"),
    ]),
    "replica-symmetry": ("Replica symmetry", [
        ("p", "Draw two configurations σ<sup>1</sup>, σ<sup>2</sup> independently from the same Gibbs measure: "
              "two <i>replicas</i>. The system is replica-symmetric when their overlap q concentrates on a single value "
              "as N → ∞. The free energy is then a formula in that one number q (the 'RS solution'). "
              "When q stays random, symmetry is broken: see replica symmetry breaking."),
        ("eq", r"$q_{12}=\frac{1}{N}\sum_{i=1}^N\sigma^1_i\sigma^2_i\ \longrightarrow\ q\quad\mathrm{(RS)}$"),
    ]),
    "interpolation": ("Guerra's interpolation", [
        ("p", "A proof technique: build a path H<sub>t</sub>, t ∈ [0, 1], from the hard model (t = 1) to a solvable one "
              "(t = 0) and control the derivative of the free energy along the path. Guerra used it in 2003 to show "
              "that the Parisi formula is an upper bound for the SK free energy; Talagrand completed the proof."),
        ("eq", r"$\varphi(t)=\frac{1}{N}\mathbb{E}\log\sum_\sigma e^{-\beta H_t(\sigma)},\qquad \varphi(1)=\varphi(0)+\int_0^1\varphi'(t)\,dt$"),
    ]),
    "poisson-dirichlet": ("Poisson–Dirichlet distribution", [
        ("p", "A random way to break a stick of length 1 into infinitely many pieces, a few large and many tiny. "
              "In the frozen REM, the Gibbs weights of the lowest-energy configurations, ranked by size, follow this "
              "law with parameter m = β<sub>c</sub>/β ∈ (0, 1): a smaller m (lower temperature) means fewer, larger "
              "pieces. It comes from normalising the points of a Poisson process with intensity x<sup>−1−m</sup> dx."),
        ("eq", r"$w_k=\frac{x_k}{\sum_j x_j},\qquad (x_k)\ \mathrm{Poisson\ process\ on\ }(0,\infty)\ \mathrm{with\ intensity}\ m\,x^{-1-m}dx$"),
    ]),
    "rsb": ("Replica symmetry breaking", [
        ("p", "Parisi's description of the low-temperature phase of spin glasses. Two independent configurations "
              "drawn from the Gibbs measure (replicas) can have different overlaps with positive probability; the "
              "distribution of the overlap is the order parameter. One step of breaking (1RSB) is the REM picture."),
    ]),
    "parisi": ("Parisi formula", [
        ("p", "The free energy of mean-field spin glasses written as an <b>infimum</b> over a functional order "
              "parameter (a distribution function on [0, 1]). For the REM-type models of this project the same "
              "infimum appears as the Lagrangian dual of a supremum over measures with an entropy budget."),
        ("eq", r"$\sup_{\nu:\,H(\nu|\mu)\leq\log 2}\left\{\Phi\!\left(\int\varphi^2d\nu\right)+\int f\,d\nu-H(\nu|\mu)\right\}=\inf_{(q,m)}P(q,m)$"),
    ]),
}

_LINK = re.compile(r"\[\[([a-z0-9\-]+)(?:\|([^\]]+))?\]\]")


def linkify(text: str, color: str) -> str:
    """Turn [[term|shown]] into rich-text links handled by the glossary dialog."""
    def rep(m):
        key = m.group(1)
        shown = m.group(2)
        if not shown:
            title = ENTRIES.get(key, (key,))[0]
            proper = title.split()[0] in {"Curie–Weiss", "Hopfield", "Sherrington–Kirkpatrick", "Random",
                                          "Parisi", "Hebb's", "Legendre", "Gibbs", "Glauber", "Guerra's",
                                          "Poisson–Dirichlet"}
            shown = title if proper else title[0].lower() + title[1:]
        return f"<a href='gloss:{key}' style='color:{color}; text-decoration:none;'>{shown}</a>"
    return _NU.sub(_NU_HTML, _LINK.sub(rep, text))


# In the sans-serif interface font the Greek ν is indistinguishable from a Latin v,
# so it is always set in a serif italic, as in a formula.
_NU = re.compile("ν")
_NU_HTML = "<span style='font-family:\"STIX Two Math\",\"Cambria Math\",\"Times New Roman\",serif; font-style:italic;'>ν</span>"


@lru_cache(maxsize=256)
def formula_png(latex: str, size: int = 15) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import mathtext
    from matplotlib.font_manager import FontProperties
    buf = io.BytesIO()
    mathtext.math_to_image(latex, buf, prop=FontProperties(size=size), dpi=220, format="png")
    return buf.getvalue()
