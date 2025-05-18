import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

plt.style.use('science')
plt.rcParams['figure.dpi'] = 300

plt.no_ticks = lambda: plt.xticks() and plt.yticks()  # type: ignore

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

__all__ = ['colors', 'plt']
