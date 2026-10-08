# Capability status legend

Every capability table in this documentation set uses these labels. They describe
what has actually been executed, not what the code intends to do.

| Label | Meaning |
|---|---|
| **REAL** | Executed against the real external tool on a developer machine, and asserted in the test suite. |
| **FIXTURE-BASED** | Exercised against recorded real output (a captured ORCA log, a recorded HTTP response) or against an analytic problem with a known answer. Correct behaviour is proven; the live integration is not re-proven on every run. |
| **MOCKED** | Exercised only against a stub the engine itself defines. Proves the code path, proves nothing about the external system. |
| **OPTIONAL** | Requires a Python package that is not a hard dependency. Absent, the code returns `UNSUPPORTED` — it does not guess and does not crash. |
| **REQUIRES-LOCAL-SOFTWARE** | Needs an external executable (GROMACS, ORCA, PLUMED) that pip does not install. |
| **REQUIRES-CREDENTIALS** | Needs an account or API key that the repository does not and must not contain. |
| **REQUIRES-EXPERT-DECISION** | Deliberately refuses to choose for you. The engine will not pick a force field, a level of theory, a reaction coordinate, or an acceptance tolerance on your behalf. |
| **NOT IMPLEMENTED** | Absent. See [ROADMAP.md](ROADMAP.md). |

A capability can carry more than one label. `REAL` and `REQUIRES-LOCAL-SOFTWARE`
together means: it genuinely runs, and it genuinely needs the tool installed.

Two labels are deliberately *not* interchangeable:

* **FIXTURE-BASED is not weaker than REAL for numerics.** WHAM validated against an
  analytic harmonic PMF is a stronger check than WHAM run on a real trajectory whose
  right answer nobody knows.
* **FIXTURE-BASED is much weaker than REAL for integration.** A recorded ORCA log
  cannot tell you whether the engine invokes ORCA correctly.
