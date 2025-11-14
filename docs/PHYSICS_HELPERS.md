# Physics helper reference

This note summarizes the equations implemented in `alchemi.physics`, the unit
conventions assumed throughout the helpers, and common pitfalls to avoid when
working with radiance, reflectance, and continuum analytics. Each section
links back to the helper's Python docstring and shows a short numerical
example.

## Unit conventions

| Quantity | Symbol | Units in helpers | Notes |
| --- | --- | --- | --- |
| Wavelength | $\lambda$ | nanometres (nm) | Convert to metres when applying Planck's law. |
| Wavenumber | $\tilde{\nu}$ | cm$^{-1}$ | Only used in some datasets; convert with $\tilde{\nu} = 10^7/\lambda_{\text{nm}}$. |
| Spectral radiance | $L_\lambda$ | W·m$^{-2}$·sr$^{-1}$·nm$^{-1}$ | Internally scaled to per-metre by multiplying by $10^{9}$. |
| Brightness temperature | $T$ | Kelvin (K) | Always returned in Kelvin. |
| Hemispherical solar irradiance | $E_0$ | W·m$^{-2}$·nm$^{-1}$ | Input for SWIR conversion helpers. |
| Surface reflectance | $R$ | unitless | Clipped to $[0, 1.5]$ in `radiance_to_reflectance`. |
| Atmospheric transmittance | $\tau$ | unitless | Bulk two-way transmittance term. |
| Path radiance | $L_{\text{path}}$ | W·m$^{-2}$·sr$^{-1}$·nm$^{-1}$ | Added to SWIR radiance model. |

## Planck-law conversions

The helpers in [`alchemi.physics.planck`](../src/alchemi/physics/planck.py) use
SI constants $h = 6.62607015\times10^{-34}$ J·s,
$c = 2.99792458\times10^{8}$ m·s$^{-1}$, and
$k_B = 1.380649\times10^{-23}$ J·K$^{-1}$.
All functions expect wavelength in nanometres and convert to metres internally.

### Spectral radiance → brightness temperature

[`radiance_to_bt_K`](../src/alchemi/physics/planck.py) implements the inverse of
Planck's law:

$$
T = \frac{h c}{\lambda k_B}\left[\ln\left(1 + \frac{2 h c^2}{L_\lambda^{(m)}\,\lambda^5}\right)\right]^{-1},
$$

where $L_\lambda^{(m)} = 10^{9} L_\lambda$ is the radiance converted from
per-nanometre to per-metre. Radiances $\leq 0$ return $0$ K by design.

*Example.* At $\lambda = 11\,\mu$m ($11000$ nm) with
$L_\lambda = 9.573\times10^{-3}$ W·m$^{-2}$·sr$^{-1}$·nm$^{-1}$, the helper
returns $T \approx 300$ K, matching the input brightness temperature.

### Brightness temperature → spectral radiance

[`bt_K_to_radiance`](../src/alchemi/physics/planck.py) applies the forward form
of Planck's law:

$$
L_\lambda^{(m)} = \frac{2 h c^2}{\lambda^5}\left(\exp\left(\frac{h c}{\lambda k_B T}\right) - 1\right)^{-1}, \qquad
L_\lambda = 10^{-9} L_\lambda^{(m)}.
$$

The exponent is clipped to avoid overflow, and the helper returns radiance in
W·m$^{-2}$·sr$^{-1}$·nm$^{-1}$. Using the previous example, $T = 300$ K at
$\lambda = 11000$ nm yields
$L_\lambda \approx 9.573\times10^{-3}$ W·m$^{-2}$·sr$^{-1}$·nm$^{-1}$.

## SWIR reflectance ↔ radiance conversion

[`reflectance_to_radiance`](../src/alchemi/physics/swir.py) and
[`radiance_to_reflectance`](../src/alchemi/physics/swir.py) implement the
single-scattering linear model used in short-wave infrared (SWIR) atmospheric
correction:

$$
L = \tau \frac{E_0 \cos\theta}{\pi} R + L_{\text{path}}, \qquad
R = \frac{L - L_{\text{path}}}{\tau \frac{E_0 \cos\theta}{\pi}},
$$

where $\theta$ is the solar zenith angle.

*Example.* With $R = 0.3$, $E_0 = 1700$ W·m$^{-2}$·nm$^{-1}$,
$\cos\theta = 0.7$, $\tau = 0.85$, and
$L_{\text{path}} = 0.02$ W·m$^{-2}$·sr$^{-1}$·nm$^{-1}$, the forward helper
returns $L \approx 96.61$ W·m$^{-2}$·sr$^{-1}$·nm$^{-1}$. Inverting recovers the
original reflectance to machine precision.

## Continuum removal and band depth

[`continuum_remove`](../src/alchemi/physics/swir.py) builds a linear continuum
between two anchor wavelengths $(\lambda_L, R_L)$ and $(\lambda_R, R_R)$:

$$
C(\lambda) = R_L + \frac{R_R - R_L}{\lambda_R - \lambda_L} (\lambda - \lambda_L),
\qquad R_{\text{cont}}(\lambda) = \frac{R(\lambda)}{\max(C(\lambda), \epsilon)}.
$$

[`band_depth`](../src/alchemi/physics/swir.py) evaluates the continuum-removed
reflectance at the centre wavelength $\lambda_C$ and reports
$BD = 1 - R_{\text{cont}}(\lambda_C)$.

*Example.* For a 2000–2500 nm window with anchors $R(2000) = 0.40$ and
$R(2500) = 0.34$, continuum removal produces $R_{\text{cont}}(2250) \approx 0.824$.
The resulting band depth is $BD \approx 0.176$.

## Common pitfalls

* **Unit scaling.** Planck helpers expect wavelengths in nanometres and radiance
  in W·m$^{-2}$·sr$^{-1}$·nm$^{-1}$. Forgetting the $10^9$ conversion between
  per-metre and per-nanometre causes order-of-magnitude errors.
* **Radiance vs. brightness temperature.** Brightness temperature is not a
  physical temperature—do not mix it with kinetic temperatures without checking
  emissivity assumptions. Prefer `radiance_to_bt_K` only when instrument
  calibration is radiometric.
* **Solar irradiance inputs.** Ensure $E_0$ matches the wavelength grid and is
  expressed per-nanometre; per-micrometre tables must be scaled by $10^3$.
* **Clip behaviour.** `radiance_to_reflectance` clips denominators below
  $10^{-12}$ and reflectance above $1.5$ to reduce numerical blow-ups.
* **Continuum anchors.** `continuum_remove` uses `np.searchsorted`, so anchors
  should align with the wavelength grid to avoid interpolating the wrong
  neighbours.
