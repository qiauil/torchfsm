# TorchFSM 0.0.6 Release Notes

## Feature Addition
* Add `Advection` Operator (81f4c96a06d40d49f2db4a32faa48c67239a3cfd).
* Add `functional_fourier_series`, `functional_energy_spectrum`,`random_power_law_energy_spectrum` and `random_diffused_noise` field function (dc1436b59dbf1fec3224c4b9bd8525d1310d5a9e,3d81f92611884c8897d42dbaeaae4ad003dd2654,1a20493d86bc4309185c5386b1defce44ea90e2d,096fbea07f39c10d3926566d91932519374c7e0d).
* Add `compare_error_field`,`compare_error_traj`, `compare_error_field_slice`, `compare_error_traj_slice`, `compare_error_traj_frame`, `compare_error_traj_frame_slice`, `plot_field_slice`, `plot_traj_frame_slice` plot functions (6b76362e097862bcb74b8ca1f298538ed2ffe923).
* Add `bf_vector_norm` function to `BroadcastedFFTFrequency` class (b147ae15a2b23aab6607a865c1fef79a8a47b67f).
* Add `LuminanceAlpha` transparency and allow to set different alpha value for `ZigzagAlpha` (bbcefc90563304252bff90a6b1b8f608780e3ca1).
* Add add traj_slices and field_slices func to the utils (6b76362e097862bcb74b8ca1f298538ed2ffe923).
* Add `test_sim_dt` function (62d379937d9bc2e94b26ad233a11a7546a0e9618).
* Add `collect_energy_spectrum` function (b47c54d1b12df9c5368ff4239a8bf6ca948569c5,f359b5f2ba8955c7e8da2a34025f8971f96fd391).
* Add more normalization modes for field functions (aca2b651ca4987ae4f9636f3821c5bde83c161e5).
* Support `rasterized` to plots (ecf1bbe28a9eceb0aba07868f96ccd7dd52644c1).
* Support to show coordinates for 3D plots (85f063aa703369f990889a2060d0f1b0701306bb).
* Support to set default integrator (6b334c159bc6f950b32bae40833bad5457ea0618).
* Support to reuse the traj recorders for different operators (1d118c1eebe8edd28eb01f24bcccdc9d192d4415).

## Feature Change
* Set `use_real_map` default to True when plotting 3D data (d0b9f472001a0cac389cb804a7186ecd65c6dc02).
* Rename `plot_3d_traj_slices` function to `plot_traj_slice`(6b76362e097862bcb74b8ca1f298538ed2ffe923).
* Rename param name `hide_batch_name_for_single_plot` to `hide_batch_channel_name_for_single_plot` (f52b69e12a51edaf810c23f1b708e7d34c51b395).
* Remove the angle and magnitude options for `truncated_fourier_series` and directly generate smooth filed from noise (321b6af88c0896534be621ee08c9cf6ee6120297).
* Change default `diffusion_coefs` values for `diffused_noise` (3a447699ca492312b831917d15d70fdfae9886d7).
* Change default plot style (6cac3b1a4cd113f3b1bfae13fb2ecb6c43b30bb1). 
* Optimize the size policy for plots (c00d3d89c5cff2b3c4eb576ca33bee7a38576b7b,82d58634223832429c8b8d0ce1d689b825813af4).


## Bugs Fixing
* Fix the duplicated alpha channel problem when calling alpha functions (7409451d0e56b37a47ad45386f80fe82a718019b).
* Fix the incorrect colorbar range moving when plot traj (20cb65e51dd01225e3daf8c01c4018d23c9e8da7).
* Fix the incorrect equation for `SwiftHohenberg` (7114a85a919f2b78cae704c05fcf69db961f0ac3).
* Minor bug fixes in plot functions (3e42ffdf76c1b3e2a2861321eb2b1a7830cfe294,38d177802ce1c94e3e3aabb9985859c6c735b531,5a86b38dacef6e5d9520f10dcd79600e344c95a2).
* Fix the trajectory issue in recorders (155cb9956e3d50715e522d1ef7cf7d247767f8ea,4f0882e0e0e5042277eaf9e85fa87513cff4f77b,5469cb4a3774219f9940a78c2330fa9462b05fcd).

# TorchFSM 0.0.5 Release Notes

## Feature Addition
* Add `Dispersion`, `HyperDiffusion`, `Leray`, `ChannelWisedDiffusion`, `GrayScottSource` operators.
* Add `KPPFisher`, `SwiftHohenberg`, `GrayScott` equations.
* Add `random_gaussian_blobs`, `truncated_fourier_series_custom_filter`, `random_truncated_fourier_series` field function.
* Add `plot_traj_frames` and `plot_3d_traj_slices` plot functions.
* Add error notation (ce886355c7f416568ed6a2870d52eb891a3ab50a).
* Add `normalized_low_pass_filter` function for mesh (5f945bcb9d635905095f51fd6f828402a4df98d7).
* Allow to check nan values during integrations (89669aa86df632292d485d5881a1e70e8096ebc9,89669aa86df632292d485d5881a1e70e8096ebc9).
* Introduce `normalize_mode` to fields (ab0b899fe11e3d65be54a48c63883c6b28fdab11).
* Allow to initialize operator directly from `LinearCoef` and `NonlinearFunc` (8abc939dd5691810026038d1991994f42da8bda0).
* Allow to return frame indices for frame select functions (adfb4cae620095721aa763f55aa1cf398237e574).
* Allow real-time ifft for `CPURecorder` (14256f56c097cd8a54a1cfb45ca555c52777ee88,8ba561a9419bcc3df814499c4418c60731589eee).
* Allow to stop simulation from recorder (8ba561a9419bcc3df814499c4418c60731589eee).

## Performance Improvement
* Change `Convection` operator to a more memory and speed-efficient version (2e996775e356e21945c2a13467a9c922055ea8ef).
* Remove lru cache to avoid memory leakage when running the simulation multiple times (f1f55b356bb461b6f47dd38927127046540871f1)

## Feature Change
* Modify the logic for recording the initial conditions ( 494d74a9232f656ea2b80602f29c3381e525e438)
* Update `KdV` equation with `Dispersion` operator to support high-dimensional simulation (1ff30c0f52b555f06031e5f143597af8a68a3708).
* Make saved plota  tight layout by default (5dc9ecb0a2a82908c6ea144a8062baa76bdad2be).

## Bugs Fixing
* Fix error raising issue of `_KSConvectionGenerator` (77fcb0387624095f2612e9da87a65ee9ddf2668d) 
* Fix None dict error when setting `de_aliasing_rate` (978a103328a654f77d355d5a2c2c6a3a0f1c36f1).
* Fix type convert issue in traj postprocess funcs (5e4757597b65102c7764ab64822f518a62b14543).
* Fix cmap error for rendering 3d field (917653ac15676dbec13f2cfeecf273308787ca4a)

Update by @qiauil 

# TorchFSM 0.0.4 Release Notes
## Performance Improvement
* Add garbage clean to optimize memory usage in 5815dc09d46a39b39b4e5b46387e32cd895caf08
* Make LR in ETDRK integrator a non-attribute variable to optimize memory performance in 74744e77c13b6390cea56e4ccbaa4e66cd8dc0a0
* Add standard ETDRK integrator and rename the original integrator as SETDRK; Allow CPU cache when building the SETDRK integrator to save memory in bb2612a71e38920654b2fd71d1d1664d116b17ce

## Functional Update
* Make normalization of `diffused_noise` batch-wised in cbde32fab6622b7cdd3a3b4ff3dec9faa2e1e46b
* Make `unit_variance` and `unit_magnitude` exclusive in diffused_noise in aeace8d092e50fc2188d8364655c6f74871745ce
* Add `RandomBatchWisedRecorder` in 6d49ff36db8a79da958cdfba89b568cc479c1e0f
* Add `print_gpu_memory function` in 453da36866d1395bf197999d736ff95a77a41ae2
* Add an absolute low-pass filter for operators in e62f827f75d1e27e6628c481a746d0f1a0874632
* Add `truncated_fourier_series` as a field function in 050fbf2925ccdcee692a361cafbe88fb2c87836d

## Name Changes
* Typo fix in f0d6fd8ace001100bc355256df43b3747c95ad45
* Change parameter name `n_batch` to `batch_size` in d5d8f779bc51bdf6c7aa56a0bf1b2d8c01da9584
* Change parameter name `num_circle_point` to `n_integration_points` and `circle_radius` to `integration_radius` for ETDRK intergrators in 0f1497765a1922f79c2861cd3aa1af9129f513c0

Update by @qiauil 