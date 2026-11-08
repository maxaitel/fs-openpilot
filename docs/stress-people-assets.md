# Stress-scene people assets

The stress scene uses three adult characters from NVIDIA's Isaac Sim 5.1 public asset collection. They are `F_Business_02`, `female_adult_police_02`, and `male_adult_construction_03`. The source USD files and their base-color, normal, and ORM textures are pinned by byte length and SHA-256 in [the lock file](../isaac_eval/stress_people.lock.json). Downloads use the public NVIDIA S3 URLs recorded there.

These are NVIDIA assets, not CC0 assets. Review the [Isaac Sim 5.1 licensing page](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/licenses-isaac-sim.html) and [NVIDIA Isaac Sim Additional Software and Materials License](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/license-isaac-sim-additional.html) before use or redistribution. Public download access does not itself grant a separate redistribution license. The source and prepared binaries are local caches and are ignored by Git.

To prepare the assets on a machine with Python, `usd-core`, and NumPy installed, run from the repository root:

```sh
python scripts/prepare_stress_people.py
```

The script downloads missing files into `.work/stress/assets-probe/source`, verifies each against the lock, and writes `isaac_eval/assets/stress_people/`. No Isaac Sim runtime or GPU is required. It fails on mismatched downloads, unexpected source rigs, non-identity mesh transforms, invalid skin weights, or missing texture dependencies. A partial or corrupted cached download is replaced and checked before use.

Each `<asset-id>/person.usd` has default prim `/Person`, Z up, and one meter per unit. The USD contains static meshes with per-vertex normals and `UsdPreviewSurface` materials. The texture files are unchanged. Deterministic per-mesh `UsdUVTexture` base-color scales tint selected garments, while skin is untinted. Business clothing uses blue jeans, a navy shirt, and a muted vest. The construction worker has blue jeans, an orange vest, and a yellow hardhat. The police character has dark navy pants, a light blue shirt, and a yellow safety vest. The construction hardhat and reflective strip meshes are explicitly made visible. The roughness input uses the green channel of the original ORM texture. The animation skeleton, MDL shaders, and physics schemas are not included. The script lowers both upper arms and bends the elbows from the source rest pose, then bakes the skinning into the vertex positions and normals. The three custom poses vary slightly, with upper arms lowered to approximately 78–80 degrees and elbows bent by 8–10 degrees around the existing rig axes. The ground is approximately `z=0`.

`manifest.json` records the prepared USD SHA-256, mesh counts, local texture paths and hashes, source hashes and byte lengths, posed joint positions, pose angles, per-mesh base-color scales, explicitly enabled parts, and measured bounds in meters. Texture paths in the manifest are relative to their asset directory. Keep each USD with its `textures/` folder when copying. A stage can reference the default prim and flatten its composed root, or an importer can copy the meshes and materials while resolving the texture asset paths against the original USD directory. Verify the resulting bundle's dependencies after either operation.

The prepared bounds measured in the local generation were approximately 1.61 m, 1.69 m, and 1.78 m tall for the business, police, and construction characters respectively. These checks and a projected silhouette inspection establish that the meshes are standing with lowered arms. They do not replace a rendered Isaac Sim scene review.
