---
schema_version: 1
display_name: Rigged Cube
source:
  primary: source/cube_rigged.fbx
  animations: ["source/animations/*.fbx"]
  license: "CC0 — synthetic fixture"
texture:
  prompt_mode: text+style
  seed: 4242
prompt:
  include_sections: [base_prompt, material_notes]
  template: "{base_prompt}\n\n{model_prompt}\n\n{material_notes}"
tags: [fixture, rigged]
---

## prompt
A small wooden crate cube with brass corner trims, warm oak planks with painted grain, soft
bevelled edges, a single stencilled mark on one face.

## notes
Placeholder FBX bytes; replaced with a real rigged cube by the fixtures work package.
