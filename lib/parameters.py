# Template to take Google colab parameters and and generate a parameters.py file with python type hinting 
# and scoping based methods they're instantiated in.
# 'parameters.py' will be automatically generated. Do not edit -- rather, rerun the file "extract_colab_parameters.py"
# This is from a jinja2 template.
# example input file: Deforum_Stable_Diffusion.ipynb

class ChangeRecorder:
# This class is used to record changes to the parameters. It is used to generate a diff of the parameters
# from initialisation to customisation. Works best with objects that have a default value in __init__ as ours do 
    def __init__(self):
        self._changed_attributes = {}

    def __setattr__(self, name, value):
        if hasattr(self, name):
            self._changed_attributes[name] = value
        super().__setattr__(name, value)

    def get_changed_attributes(self):
        return self._changed_attributes.copy()

    def get_filesystem_friendly_changed_attributes(self):
        filesystem_friendly_attributes = []
        for attribute_name, attribute_value in self._changed_attributes.items():
            # Replace any invalid characters in the attribute name with underscores
            filesystem_friendly_name = attribute_name.replace(":", "_")
            # Replace any invalid characters in the attribute value with an empty string
            filesystem_friendly_value = str(attribute_value).replace(":", "").replace("--", "")
            # Concatenate the file-system friendly attribute name and value using a separator
            filesystem_friendly_string = f"{filesystem_friendly_name}:{filesystem_friendly_value}"
            filesystem_friendly_attributes.append(filesystem_friendly_string)
        # Concatenate all the transformed attribute strings using a separator
        return "--".join(filesystem_friendly_attributes)
        
class Root(ChangeRecorder):
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
class DeforumArgs(ChangeRecorder):
  def __init__(self, **kwargs):
      super().__init__()
      self.override_settings_with_file: bool = kwargs.get("override_settings_with_file", False)
      self.settings_file = kwargs.get("settings_file", 'custom')  # custom, 512x512_aesthetic_0.json, 512x512_aesthetic_1.json, 512x512_colormatch_0.json, 512x512_colormatch_1.json, 512x512_colormatch_2.json, 512x512_colormatch_3.json
      self.custom_settings_file: str = kwargs.get("custom_settings_file", "/content/drive/MyDrive/Settings.txt")
      self.W = kwargs.get("W", 768)
      self.H = kwargs.get("H", 512)
      self.bit_depth_output: str = kwargs.get("bit_depth_output", "8")  # 8, 16, 32
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
class DeforumAnimArgs(ChangeRecorder):
  def __init__(self, **kwargs):
      super().__init__()
      self.animation_mode: str = kwargs.get("animation_mode", "2D")  # None, 2D, 3D, Video Input, Interpolation
      self.max_frames: float = kwargs.get("max_frames", 270)
      self.border: str = kwargs.get("border", "replicate")  # wrap, replicate
      self.angle: str = kwargs.get("angle", "0:(0)")
      self.zoom: str = kwargs.get("zoom", "0: (1.22), 1: (1.06), 2: (1.04), 3: (1.18), 4: (1.06), 5: (1.04), 6: (1.16), 7: (1.05), 8: (1.04), 9: (1.17), 10: (1.15), 11: (1.04), 12: (1.18), 13: (1.05), 14: (1.04), 15: (1.14), 16: (1.06), 17: (1.04), 18: (1.15), 19: (1.05), 20: (1.04), 21: (1.16), 22: (1.16), 23: (1.04), 24: (1.18), 25: (1.05), 26: (1.04), 27: (1.15), 28: (1.06), 29: (1.04), 30: (1.18), 31: (1.05), 32: (1.04), 33: (1.16), 34: (1.14), 35: (1.04), 36: (1.19), 37: (1.05), 38: (1.04), 39: (1.18), 40: (1.06), 41: (1.04), 42: (1.20), 43: (1.05), 44: (1.04), 45: (1.15), 46: (1.13), 47: (1.04), 48: (1.16), 49: (1.05), 50: (1.04), 51: (1.18), 52: (1.06), 53: (1.04), 54: (1.20), 55: (1.05), 56: (1.04), 57: (1.15), 58: (1.13), 59: (1.04), 60: (1.19), 61: (1.06), 62: (1.04), 63: (1.17), 64: (1.06), 65: (1.04), 66: (1.18), 67: (1.05), 68: (1.04), 69: (1.15), 70: (1.17), 71: (1.04), 72: (1.14), 73: (1.05), 74: (1.04), 75: (1.19), 76: (1.06), 77: (1.04), 78: (1.21), 79: (1.05), 80: (1.04), 81: (1.16), 82: (1.14), 83: (1.04), 84: (1.15), 85: (1.06), 86: (1.04), 87: (1.19), 88: (1.06), 89: (1.04), 90: (1.20), 91: (1.05), 92: (1.04), 93: (1.15), 94: (1.13), 95: (1.04), 96: (1.22), 97: (1.05), 98: (1.04), 99: (1.21), 100: (1.05), 101: (1.04), 102: (1.20), 103: (1.05), 104: (1.04), 105: (1.14), 106: (1.14), 107: (1.04), 108: (1.18), 109: (1.05), 110: (1.04), 111: (1.17), 112: (1.05), 113: (1.04), 114: (1.23), 115: (1.05), 116: (1.04), 117: (1.19), 118: (1.14), 119: (1.04), 120: (1.17), 121: (1.05), 122: (1.04), 123: (1.17), 124: (1.05), 125: (1.04), 126: (1.19), 127: (1.04), 128: (1.04), 129: (1.17), 130: (1.13), 131: (1.04), 132: (1.16), 133: (1.05), 134: (1.04), 135: (1.17), 136: (1.05), 137: (1.04), 138: (1.20), 139: (1.04), 140: (1.04), 141: (1.14), 142: (1.13), 143: (1.04), 144: (1.20), 145: (1.05), 146: (1.04), 147: (1.16), 148: (1.05), 149: (1.04), 150: (1.19), 151: (1.05), 152: (1.04), 153: (1.14), 154: (1.11), 155: (1.04), 156: (1.19), 157: (1.05), 158: (1.04), 159: (1.19), 160: (1.05), 161: (1.04), 162: (1.16), 163: (1.04), 164: (1.04), 165: (1.15), 166: (1.13), 167: (1.04), 168: (1.04), 169: (1.04), 170: (1.04), 171: (1.04), 172: (1.04), 173: (1.04), 174: (1.54), 175: (1.24), 176: (1.05), 177: (1.43), 178: (1.21), 179: (1.06), 180: (1.37), 181: (1.25), 182: (1.05), 183: (1.85), 184: (2.02), 185: (1.06), 186: (1.55), 187: (1.25), 188: (1.06), 189: (1.44), 190: (1.19), 191: (1.06), 192: (1.51), 193: (1.20), 194: (1.06), 195: (1.50), 196: (1.22), 197: (1.50), 198: (1.49), 199: (1.21), 200: (1.06), 201: (1.44), 202: (1.22), 203: (1.06), 204: (1.47), 205: (1.20), 206: (1.05), 207: (1.75), 208: (1.82), 209: (1.06), 210: (1.52), 211: (1.23), 212: (1.06), 213: (1.52), 214: (1.21), 215: (1.06), 216: (1.46), 217: (1.26), 218: (1.06), 219: (1.47), 220: (1.21), 221: (1.49), 222: (1.45), 223: (1.21), 224: (1.06), 225: (1.52), 226: (1.21), 227: (1.06), 228: (1.43), 229: (1.24), 230: (1.06), 231: (1.83), 232: (1.80), 233: (1.05), 234: (1.55), 235: (1.20), 236: (1.05), 237: (1.41), 238: (1.23), 239: (1.06), 240: (1.42), 241: (1.22), 242: (1.06), 243: (1.50), 244: (1.24), 245: (1.56), 246: (1.50), 247: (1.20), 248: (1.05), 249: (1.42), 250: (1.21), 251: (1.05), 252: (1.39), 253: (1.22), 254: (1.06), 255: (1.74), 256: (2.04), 257: (1.06), 258: (1.45), 259: (1.19), 260: (1.05), 261: (1.48), 262: (1.22), 263: (1.06), 264: (1.46), 265: (1.14), 266: (1.04), 267: (1.12), 268: (1.04), 269: (1.04), 270: (1.04)")
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
