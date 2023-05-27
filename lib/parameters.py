# Template to take Google colab parameters and and generate a parameters.py file with python type hinting 
# and scoping based methods they're instantiated in.
# 'parameters.py' will be automatically generated. Do not edit -- rather, rerun the file "extract_colab_parameters.py"
# This is from a jinja2 template.
# example input file: Deforum_Stable_Diffusion.ipynb

        
class DeforumAnimArgs:
  def __init__(self, **kwargs):
      super().__init__()
      self.animation_mode: str = kwargs.get("animation_mode", "2D")  # None, 2D, 3D, Video Input, Interpolation
      self.max_frames: float = kwargs.get("max_frames", 10)
      self.border: str = kwargs.get("border", "replicate")  # wrap, replicate
      self.angle: str = kwargs.get("angle", "0:(0)")
      self.zoom: str = kwargs.get("zoom", "0: (1.22)")
      self.translation_x: str = kwargs.get("translation_x", "0:(0)")
      self.translation_y: str = kwargs.get("translation_y", "0:(3)")
      self.translation_z: str = kwargs.get("translation_z", "0:(0)")
      self.rotation_3d_x: str = kwargs.get("rotation_3d_x", "0:(0)")
      self.rotation_3d_y: str = kwargs.get("rotation_3d_y", "0:(0)")
      self.rotation_3d_z: str = kwargs.get("rotation_3d_z", "0:(0)")
      self.flip_2d_perspective: bool = kwargs.get("flip_2d_perspective", False)
      self.perspective_flip_theta: str = kwargs.get("perspective_flip_theta", "0:(0)")
      self.perspective_flip_phi: str = kwargs.get("perspective_flip_phi", "0:(0)")
      self.perspective_flip_gamma: str = kwargs.get("perspective_flip_gamma", "0:(0)")
      self.perspective_flip_fv: str = kwargs.get("perspective_flip_fv", "0:(53)")
      self.noise_schedule: str = kwargs.get("noise_schedule", "0: (0.02)")
      self.strength_schedule: str = kwargs.get("strength_schedule", "0: (0.70)")
      self.contrast_schedule: str = kwargs.get("contrast_schedule", "0: (1.0)")
      self.hybrid_video_comp_alpha_schedule: str = kwargs.get("hybrid_video_comp_alpha_schedule", "0:(1)")
      self.hybrid_video_comp_mask_blend_alpha_schedule: str = kwargs.get("hybrid_video_comp_mask_blend_alpha_schedule", "0:(0.5)")
      self.hybrid_video_comp_mask_contrast_schedule: str = kwargs.get("hybrid_video_comp_mask_contrast_schedule", "0:(1)")
      self.hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule: str = kwargs.get("hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule", "0:(100)")
      self.hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule: str = kwargs.get("hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule", "0:(0)")
      self.kernel_schedule: str = kwargs.get("kernel_schedule", "0: (5)")
      self.sigma_schedule: str = kwargs.get("sigma_schedule", "0: (1.0)")
      self.amount_schedule: str = kwargs.get("amount_schedule", "0: (0.2)")
      self.threshold_schedule: str = kwargs.get("threshold_schedule", "0: (0.0)")
      self.color_coherence: str = kwargs.get("color_coherence", "Match Frame 0 LAB")  # None, Match Frame 0 HSV, Match Frame 0 LAB, Match Frame 0 RGB, Video Input
      self.color_coherence_video_every_N_frames: int = kwargs.get("color_coherence_video_every_N_frames", 1)
      self.diffusion_cadence: str = kwargs.get("diffusion_cadence", "1")  # 1, 2, 3, 4, 5, 6, 7, 8
      self.use_depth_warping: bool = kwargs.get("use_depth_warping", True)
      self.midas_weight: float = kwargs.get("midas_weight", 0.3)
      self.fov: float = kwargs.get("fov", 40)
      self.padding_mode: str = kwargs.get("padding_mode", "border")  # border, reflection, zeros
      self.sampling_mode: str = kwargs.get("sampling_mode", "bicubic")  # bicubic, bilinear, nearest
      self.save_depth_maps: bool = kwargs.get("save_depth_maps", False)
      self.video_init_path: str = kwargs.get("video_init_path", "/content/video_in.mp4")
      self.extract_nth_frame: float = kwargs.get("extract_nth_frame", 1)
      self.overwrite_extracted_frames: bool = kwargs.get("overwrite_extracted_frames", True)
      self.use_mask_video: bool = kwargs.get("use_mask_video", False)
      self.video_mask_path: str = kwargs.get("video_mask_path", "/content/video_in.mp4")
      self.hybrid_video_generate_inputframes: bool = kwargs.get("hybrid_video_generate_inputframes", False)
      self.hybrid_video_use_first_frame_as_init_image: bool = kwargs.get("hybrid_video_use_first_frame_as_init_image", True)
      self.hybrid_video_motion = kwargs.get("hybrid_video_motion", 'None')  # None, Optical Flow, Perspective, Affine
      self.hybrid_video_flow_method = kwargs.get("hybrid_video_flow_method", 'Farneback')  # Farneback, DenseRLOF, SF
      self.hybrid_video_composite: bool = kwargs.get("hybrid_video_composite", False)
      self.hybrid_video_comp_mask_type = kwargs.get("hybrid_video_comp_mask_type", 'None')  # None, Depth, Video Depth, Blend, Difference
      self.hybrid_video_comp_mask_inverse: bool = kwargs.get("hybrid_video_comp_mask_inverse", False)
      self.hybrid_video_comp_mask_equalize = kwargs.get("hybrid_video_comp_mask_equalize", 'None')  # None, Before, After, Both
      self.hybrid_video_comp_mask_auto_contrast: bool = kwargs.get("hybrid_video_comp_mask_auto_contrast", False)
      self.hybrid_video_comp_save_extra_frames: bool = kwargs.get("hybrid_video_comp_save_extra_frames", False)
      self.hybrid_video_use_video_as_mse_image: bool = kwargs.get("hybrid_video_use_video_as_mse_image", False)
      self.interpolate_key_frames: bool = kwargs.get("interpolate_key_frames", False)
      self.interpolate_x_frames: float = kwargs.get("interpolate_x_frames", 4)
      self.resume_from_timestring: bool = kwargs.get("resume_from_timestring", False)
      self.resume_timestring: str = kwargs.get("resume_timestring", "20220829210106")
class DeforumArgs:
  def __init__(self, **kwargs):
      super().__init__()
      self.override_settings_with_file: bool = kwargs.get("override_settings_with_file", False)
      self.settings_file = kwargs.get("settings_file", 'custom')  # custom, 512x512_aesthetic_0.json, 512x512_aesthetic_1.json, 512x512_colormatch_0.json, 512x512_colormatch_1.json, 512x512_colormatch_2.json, 512x512_colormatch_3.json
      self.custom_settings_file: str = kwargs.get("custom_settings_file", "/content/drive/MyDrive/Settings.txt")
      self.W = kwargs.get("W", 512)
      self.H = kwargs.get("H", 512)
      self.bit_depth_output: str = kwargs.get("bit_depth_output", "8")  # 8, 16, 32
      self.general_style: str = kwargs.get("general_style", " in watercolor, cosy, trending on artstation")
      self.seed = kwargs.get("seed", -1)
      self.sampler = kwargs.get("sampler", 'euler_ancestral')  # klms, dpm2, dpm2_ancestral, heun, euler, euler_ancestral, plms, ddim, dpm_fast, dpm_adaptive, dpmpp_2s_a, dpmpp_2m
      self.steps = kwargs.get("steps", 50)
      self.scale = kwargs.get("scale", 7)
      self.ddim_eta = kwargs.get("ddim_eta", 0.0)
      self.save_samples: bool = kwargs.get("save_samples", True)
      self.save_settings: bool = kwargs.get("save_settings", True)
      self.display_samples: bool = kwargs.get("display_samples", True)
      self.save_sample_per_step: bool = kwargs.get("save_sample_per_step", False)
      self.show_sample_per_step: bool = kwargs.get("show_sample_per_step", False)
      self.prompt_weighting: bool = kwargs.get("prompt_weighting", True)
      self.normalize_prompt_weights: bool = kwargs.get("normalize_prompt_weights", True)
      self.log_weighted_subprompts: bool = kwargs.get("log_weighted_subprompts", False)
      self.n_batch = kwargs.get("n_batch", 1)
      self.batch_name: str = kwargs.get("batch_name", "")
      self.filename_format = kwargs.get("filename_format", '{timestring}_{index}_{prompt}.png')  # {timestring}_{index}_{seed}.png, {timestring}_{index}_{prompt}.png
      self.seed_behavior = kwargs.get("seed_behavior", 'iter')  # iter, fixed, random, ladder, alternate
      self.seed_iter_N: int = kwargs.get("seed_iter_N", 1)
      self.make_grid: bool = kwargs.get("make_grid", False)
      self.grid_rows = kwargs.get("grid_rows", 2)
      self.use_init: bool = kwargs.get("use_init", False)
      self.strength: float = kwargs.get("strength", 0.1)
      self.init_image: str = kwargs.get("init_image", "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg")
      self.use_mask: bool = kwargs.get("use_mask", False)
      self.mask_file: str = kwargs.get("mask_file", "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg")
      self.invert_mask: bool = kwargs.get("invert_mask", False)
      self.mask_brightness_adjust: float = kwargs.get("mask_brightness_adjust", 1.0)
      self.mask_contrast_adjust: float = kwargs.get("mask_contrast_adjust", 1.0)
      self.overlay_mask: bool = kwargs.get("overlay_mask", True)
      self.mask_overlay_blur: float = kwargs.get("mask_overlay_blur", 5)
      self.mean_scale: float = kwargs.get("mean_scale", 0)
      self.var_scale: float = kwargs.get("var_scale", 0)
      self.exposure_scale: float = kwargs.get("exposure_scale", 0)
      self.exposure_target: float = kwargs.get("exposure_target", 0.5)
      self.colormatch_scale: float = kwargs.get("colormatch_scale", 0)
      self.colormatch_image: str = kwargs.get("colormatch_image", "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png")
      self.colormatch_n_colors: float = kwargs.get("colormatch_n_colors", 4)
      self.ignore_sat_weight: float = kwargs.get("ignore_sat_weight", 0)
      self.clip_name = kwargs.get("clip_name", 'ViT-L/14')  # ViT-L/14, ViT-L/14@336px, ViT-B/16, ViT-B/32
      self.clip_scale: float = kwargs.get("clip_scale", 0)
      self.aesthetics_scale: float = kwargs.get("aesthetics_scale", 0)
      self.cutn: float = kwargs.get("cutn", 1)
      self.cut_pow: float = kwargs.get("cut_pow", 0.0001)
      self.init_mse_scale: float = kwargs.get("init_mse_scale", 0)
      self.init_mse_image: str = kwargs.get("init_mse_image", "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg")
      self.blue_scale: float = kwargs.get("blue_scale", 0)
      self.gradient_wrt = kwargs.get("gradient_wrt", 'x0_pred')  # x, x0_pred
      self.gradient_add_to = kwargs.get("gradient_add_to", 'both')  # cond, uncond, both
      self.decode_method = kwargs.get("decode_method", 'linear')  # autoencoder, linear
      self.grad_threshold_type = kwargs.get("grad_threshold_type", 'dynamic')  # dynamic, static, mean, schedule
      self.clamp_grad_threshold: float = kwargs.get("clamp_grad_threshold", 0.2)
      self.clamp_start = kwargs.get("clamp_start", 0.2)
      self.clamp_stop = kwargs.get("clamp_stop", 0.01)
      self.grad_inject_timing = kwargs.get("grad_inject_timing", list(range(1,10)))
      self.cond_uncond_sync: bool = kwargs.get("cond_uncond_sync", True)
class Root:
  def __init__(self, **kwargs):
      super().__init__()
      self.models_path: str = kwargs.get("models_path", "models")
      self.configs_path: str = kwargs.get("configs_path", "configs")
      self.output_path: str = kwargs.get("output_path", "outputs")
      self.mount_google_drive: bool = kwargs.get("mount_google_drive", True)
      self.models_path_gdrive: str = kwargs.get("models_path_gdrive", "/content/drive/MyDrive/AI/models")
      self.output_path_gdrive: str = kwargs.get("output_path_gdrive", "/content/drive/MyDrive/AI/StableDiffusion")
      self.map_location = kwargs.get("map_location", 'cuda')  # cpu, cuda
      self.model_config = kwargs.get("model_config", 'v1-inference.yaml')  # custom, v2-inference.yaml, v2-inference-v.yaml, v1-inference.yaml
      self.model_checkpoint = kwargs.get("model_checkpoint", 'Protogen_V2.2.ckpt')  # custom, v2-1_768-ema-pruned.ckpt, v2-1_512-ema-pruned.ckpt, 768-v-ema.ckpt, 512-base-ema.ckpt, Protogen_V2.2.ckpt, v1-5-pruned.ckpt, v1-5-pruned-emaonly.ckpt, sd-v1-4-full-ema.ckpt, sd-v1-4.ckpt, sd-v1-3-full-ema.ckpt, sd-v1-3.ckpt, sd-v1-2-full-ema.ckpt, sd-v1-2.ckpt, sd-v1-1-full-ema.ckpt, sd-v1-1.ckpt, robo-diffusion-v1.ckpt, wd-v1-3-float16.ckpt
      self.custom_config_path: str = kwargs.get("custom_config_path", "")
      self.custom_checkpoint_path: str = kwargs.get("custom_checkpoint_path", "")
