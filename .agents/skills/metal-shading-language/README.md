# Metal Shading Language Skill

A bundled [opencode](https://opencode.ai) skill that provides spec-grounded assistance for Metal Shading Language (MSL) questions. This repo now contains the actual skill source (`SKILL.md` + `references/`) as well as a packaged `.skill` artifact.

## What this skill does

- Answers MSL questions with direct references to the Apple spec
- Helps write, explain, review, debug, and refactor `.metal` shader code
- Surfaces version constraints, GPU-family requirements, and address-space rules
- Provides code examples aligned with documented MSL constraints

## Repo layout

- `SKILL.md` - the installable skill source and trigger description
- `references/` - bundled MSL reference chapters used by the skill
- `metal-shading-language.skill` - packaged artifact for direct installation

## Coverage

| Chapter | File | What it covers |
|---|---|---|
| Front matter | `references/chapters/00-front-matter.md` | Cover, copyright, table of contents |
| 1 - Introduction | `references/chapters/01-introduction.md` | Language model, compiler flags, versioning, coordinate systems |
| 2 - Data Types | `references/chapters/02-data-types.md` | Scalar/vector/matrix types, resource types, packed types, alignment |
| 3 - Operators | `references/chapters/03-operators.md` | Operator semantics, expression edge cases |
| 4 - Address Spaces | `references/chapters/04-address-spaces.md` | `device`, `constant`, `thread`, `threadgroup`, and specialized spaces |
| 5 - Functions & Variables | `references/chapters/05-function-and-variable-declarations.md` | Entry points, attributes, stage I/O, resource bindings |
| 6 - Metal Standard Library | `references/chapters/06-metal-standard-library.md` | Built-ins, math, synchronization, SIMD-group APIs |
| 7 - Metal Performance Primitives | `references/chapters/07-metal-performance-primitives.md` | MPP tensor ops, cooperative tensors |
| 8 - Numerical Compliance | `references/chapters/08-numerical-compliance.md` | IEEE-754 deviations, NaN/INF, ULP accuracy, texture conversions |
| 9 - Appendix | `references/chapters/09-appendix.md` | Metal 3.2 additions |

## Installation

Download `metal-shading-language.skill` from this repo's releases and place it in your opencode skills directory (typically `~/.agents/skills/`), or install from source by copying `SKILL.md` and `references/` into a `metal-shading-language/` skill folder.

## About the spec content

The chapter files are OCR output from Apple's **Metal Shading Language Specification** (publicly available at [developer.apple.com](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)). They were parsed and cleaned for use as a bundled skill reference. Any OCR artifacts are unintentional — please open an issue if you spot corrupted content that affects the skill's usefulness.

## License

The original skill material in this repo, including `SKILL.md`, repository metadata, and packaging work, is released under the [MIT License](LICENSE).

The bundled chapter text in `references/chapters/` is derived from Apple's Metal Shading Language Specification via OCR cleanup. Apple retains rights in the original specification. This repository does not claim ownership of Apple's source text, and redistribution of the derived chapter files should be treated as subject to Apple's terms and any applicable law. Keep this repository private unless you have confirmed you have the rights to redistribute that content.
