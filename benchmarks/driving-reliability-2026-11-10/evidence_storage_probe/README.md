# Evidence storage planning

A frozen 200-frame NPZ test verifies exact lossless tar/zstd reconstruction, but saves only0.355 percent. The first attempt failed strict inventory because Mac tar added200AppleDouble metadata members; all200actualpayloads were exact. Version2 disables only that tar metadata. Both archives, original sources and the failed receipt remain retained.

The observed frame-size estimate for80maximum-length runs is about208GB before scenes and videos. About250GB is a planning estimate, not a strict upper bound. Actual sizes depend on scenes and controller duration. Existing Mac/Spark space is insufficient for that plan while retaining reserves and development evidence. An additional location with300GBfree was requested asynchronously;500GB is preferable. No final run is opened, and capacity must be checked again before launch. These storage measurements do not describe driving or model compute efficiency.

`v1/` and `v2/` preserve each frozen script, protocol, selection and outcome. Large compressed archives remain outside Git under `runs/evidence_compression_probe_v{1,2}`. The copy receipt binds the compact files.
