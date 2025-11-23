# KerrGeodesics
# Towards real-time visualization of the Kerr spacetime.

[![Build Status](https://github.com/ArchHem/KerrGeodesics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ArchHem/KerrGeodesics.jl/actions/workflows/CI.yml?query=branch%3Amain)

# TODO's

* Better spacial coherence (reorder warp tiles to morton-like order inside the image frame, hopefully will cause less stalls)
* Improve camera abstraction (pass a SubStruct object to the constructor function)
* Better timestepping heuretics/calibration

# Far-shot TODOs

* Pre-render kernel sampler rays
* Backed specific texture memory for the sampler (unlikely, its not the bottleneck)
* Enzyme generated pixels/frames
* Test CUDA