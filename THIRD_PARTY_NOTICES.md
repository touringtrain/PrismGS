# Third-party provenance

Training/rendering derives from V³ / VideoGS
(https://github.com/AuthorityWang/VideoGS, upstream revision
3a858e5c18d1ff2ed0527623e03581ac27c0fa81), the authors' local VideoGS
modifications, and their consolidated AceGS tree. Import hashes are recorded
under `training/`. Unmodified upstream source remains in `third_party/videogs/`.

Original Inria/Graphdeco copyright notices are retained. The referenced
research/noncommercial Gaussian Splatting license is supplied in
`licenses/gaussian-splatting-LICENSE.md`. This does not establish a license for
all V³ additions or for original PrismGS contributions; these terms still need
confirmation by the authors before a public release.

CUDA extension licenses are included with `submodules/simple-knn` and
`submodules/diff-gaussian-rasterization`; GLM includes its original copyright
and license text. Tiny CUDA NN is an external BSD-3-Clause installation dependency.
No third-party implementation is relicensed as MIT or CC by this repository.

The bundled Lumos5G trace and legacy example/media assets are subject to their
original redistribution terms, which remain to be confirmed separately.
The paper's publishing license does not automatically apply to software/data.
