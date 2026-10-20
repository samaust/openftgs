# The SH basis in `openftgs.sh`

How `openftgs.sh` meets its §2.1 row: the real SH basis that E2 evaluates, and the conversion between an RGB color and the SH DC band. The public names are a provisional decision (Q-14).

## Band order and signs

E2 requires gsplat's constants and band ordering, because the gsplat render backend evaluates colors itself from the same coefficients. `sh_basis` therefore orders the basis by band l = 0 … L and, within a band, by m = −l … l, and uses gsplat 1.5.3's signs: the real SH that keep the Condon–Shortley phase, so that, for example, the l = 1 band responds to −y, z and −x. U-6 checks the basis against an independent evaluation from the general definition, and a GPU test checks it against gsplat itself.

## Directions

`sh_basis` normalizes the directions it receives, as E2's d is a unit vector. Callers pass the vector from the camera center to the moved mean (E2, D-4) without normalizing it; choosing that vector is the render backend's job (M4).

## The SH DC band and RGB

`rgb_to_sh0` and `sh0_to_rgb` convert between a color and the SH DC band with C₀ and the offset of D-4; initialization sets the SH DC band this way (§1.1). The clamp at zero and the inactive bands of E2 belong to the render backend, not to the basis.

## Colour from the basis

`sh_to_color(sh, dirs)` is E2's colour: max(0, Σ c·Y(d) + 0.5) per channel, with the degree taken from the number of coefficients and `dirs` normalized inside. The reference backend's colours come from it, and the harness's `sh_color` query reads it (Q-204).
