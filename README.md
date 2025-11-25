# KerrGeodesics
# Towards real-time visualization of the Kerr spacetime.

![Demo](example_media/output_800x800_360frames_30fps.gif)

[![Build Status](https://github.com/ArchHem/KerrGeodesics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ArchHem/KerrGeodesics.jl/actions/workflows/CI.yml?query=branch%3Amain)

# What is KerrGeodesics.jl?

*KerrGeodesics.jl* is a package that aims to facilitiate realtime, and accurate, visualization of the Kerr spacetime. It is primary aimed for visualization and performance (i.e., for instance, time-stepping is based on heuretic), and not neccesiraly for general geodesic calculations, but it supports such in its API. 

It support both CPU and GPU backends thorugh *KernelAbstractions.jl*, and there are plans for specific dispatches for improved CPU performance through *SIMD.jl* usage.

# TODO's

* Better spacial coherence (reorder warp tiles to morton-like order inside the image frame, hopefully will cause less stalls) DONE
* Improve camera abstraction (pass a SubStruct object to the constructor function) DONE
* Better timestepping heuretics/calibration

# Far-shot TODOs

* Pre-render kernel sampler rays
* Backed specific texture memory for the sampler (unlikely, its not the bottleneck)
* Enzyme generated pixels/frames
* Test CUDA

# Literature and projects used

* Hamiltonian EoM: https://iopscience.iop.org/article/10.3847/1538-4357/abdc28
* Further EoM: https://iopscience.iop.org/article/10.3847/1538-4365/ac77ef/pdf
* Integrator choices (reduced to just RK4 for now): https://iopscience.iop.org/article/10.3847/1538-4365/aac9ca/pdf?fbclid=IwAR0pORzJb6EvCVdTIWo32F6wxhdd3_eQE_-x8afe94Y8dY_2IH_NuNcPiD0
* Camera tetrad: https://arxiv.org/pdf/1410.7775