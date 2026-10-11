# Good and Bad Tests

The examples use `mcmc_split` from spec §2.1 (`openftgs.density`, E11) and the U-13 row: n = 2 gives 1 − √(1 − σ), n stacked copies compose back to σ, 1e-10 relative in float64. Take the signature from §2.1 and every value and tolerance from the row of the test you are writing.

## Good tests

**Written from the spec row, through the public interface, against a literal worked by hand.**

```python
# SPDX-License-Identifier: Apache-2.0
import torch

from openftgs.density import mcmc_split


def test_u13_two_copies_give_one_minus_sqrt_of_survival():
    """U-13: with n = 2, σ = 0.75 becomes 1 − √0.25 = 0.5."""
    opacity = torch.tensor([0.75], dtype=torch.float64)
    scales = torch.ones(1, 3, dtype=torch.float64)
    new_opacity, _ = mcmc_split(opacity, scales, torch.tensor([2]))
    torch.testing.assert_close(
        new_opacity, torch.tensor([0.5], dtype=torch.float64), rtol=1e-10, atol=0.0
    )
```

**A property the row states, drawn by hypothesis within its domain.**

```python
from hypothesis import given, strategies as st


@given(sigma=st.floats(0.3, 0.99), n=st.integers(1, 51))
def test_u13_copies_compose_back_to_the_original_opacity(sigma, n):
    """U-13: n stacked copies at σ_new have the original opacity, 1 − (1 − σ_new)^n = σ.

    σ ≥ 0.3 keeps σ_new above the 0.005 clamp for every n ≤ 51.
    """
    opacity = torch.tensor([sigma], dtype=torch.float64)
    new_opacity, _ = mcmc_split(opacity, torch.ones(1, 3, dtype=torch.float64), torch.tensor([n]))
    torch.testing.assert_close(1 - (1 - new_opacity) ** n, opacity, rtol=1e-10, atol=0.0)
```

Characteristics:

- Cites the §3 ID and states the property in its name or docstring
- Uses the §2.1 public interface only
- Takes cases, values and tolerances from the spec row
- Survives internal refactors
- Holds one spec item: several assertions are fine when the row states several (U-1 states three), each with the row's tolerance

## Bad tests

**Tautological**: the expected value restates E11 the way the code computes it, so the test passes by construction.

```python
# BAD: the test recomputes the formula it is meant to check
def test_relocation_opacity():
    opacity = torch.rand(8, dtype=torch.float64)
    n = torch.randint(1, 6, (8,))
    new_opacity, _ = mcmc_split(opacity, torch.ones(8, 3, dtype=torch.float64), n)
    torch.testing.assert_close(new_opacity, 1 - (1 - opacity) ** (1 / n))
```

It also draws σ from (0, 1), so the 0.005 clamp silently changes the expected value for small σ and large n: random inputs need the row's domain.

**Implementation-coupled**: the test patches an internal collaborator and asserts on the call.

```python
# BAD: breaks if relocate() is restructured, though its behaviour is unchanged
def test_relocate_calls_mcmc_split(monkeypatch, model, optimizer, stats, reloc_cfg):
    calls = []
    monkeypatch.setattr(
        "openftgs.density.mcmc_split", lambda *args: calls.append(args) or args[:2]
    )
    relocate(model, optimizer, stats, reloc_cfg, generator=torch.Generator().manual_seed(0))
    assert len(calls) == 1
```

The spec checks relocation through its outcome instead: U-13 renders before and after relocating 30 dead primitives onto 10 targets, at the targets' μt, and requires PSNR ≥ 30 dB.

Red flags:

- Mocking anything inside `openftgs`
- Testing private functions
- Asserting on call counts or order
- Expected values computed by the code under test, or by the same formula
- Random inputs outside the row's stated domain
- A tolerance that isn't the row's
- Test breaks when refactoring without behavior change
