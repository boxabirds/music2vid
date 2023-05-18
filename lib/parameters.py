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
        if hasattr(self, name) and getattr(self, name) != value:
            self._changed_attributes[name] = getattr(self, name)
        super().__setattr__(name, value)

    def get_changed_attributes(self):
        return self._changed_attributes.copy()
        
class DeforumArgs(ChangeRecorder):
  def __init__(self):
      super().__init__()
      self.override_settings_with_file: bool = False
      self.settings_file = "custom"  # custom, 512x512_aesthetic_0.json, 512x512_aesthetic_1.json, 512x512_colormatch_0.json, 512x512_colormatch_1.json, 512x512_colormatch_2.json, 512x512_colormatch_3.json
      self.custom_settings_file: str = "/content/drive/MyDrive/Settings.txt"
      self.W = "768"
      self.H = "512"
      self.bit_depth_output: str = "8"  # 8, 16, 32
      self.seed = "-1"
      self.sampler = "'euler_ancestral'"  # klms, dpm2, dpm2_ancestral, heun, euler, euler_ancestral, plms, ddim, dpm_fast, dpm_adaptive, dpmpp_2s_a, dpmpp_2m
      self.steps = "50"
      self.scale = "7"
      self.ddim_eta = "0.0"
      self.save_samples: bool = True
      self.save_settings: bool = True
      self.display_samples: bool = True
      self.save_sample_per_step: bool = False
      self.show_sample_per_step: bool = False
      self.prompt_weighting: bool = True
      self.normalize_prompt_weights: bool = True
      self.log_weighted_subprompts: bool = False
      self.n_batch = "1"
      self.batch_name: str = ""
      self.filename_format = "{timestring}_{index}_{prompt}.png"  # {timestring}_{index}_{seed}.png, {timestring}_{index}_{prompt}.png
      self.seed_behavior = "iter"  # iter, fixed, random, ladder, alternate
      self.seed_iter_N: int = 1
      self.make_grid: bool = False
      self.grid_rows = "2"
      self.use_init: bool = False
      self.strength: float = 0.1
      self.init_image: str = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"
      self.use_mask: bool = False
      self.mask_file: str = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg"
      self.invert_mask: bool = False
      self.mask_brightness_adjust: float = 1.0
      self.mask_contrast_adjust: float = 1.0
      self.overlay_mask: bool = True
      self.mask_overlay_blur: float = 5
      self.mean_scale: float = 0
      self.var_scale: float = 0
      self.exposure_scale: float = 0
      self.exposure_target: float = 0.5
      self.colormatch_scale: float = 0
      self.colormatch_image: str = "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png"
      self.colormatch_n_colors: float = 4
      self.ignore_sat_weight: float = 0
      self.clip_name = "'ViT-L/14'"  # ViT-L/14, ViT-L/14@336px, ViT-B/16, ViT-B/32
      self.clip_scale: float = 0
      self.aesthetics_scale: float = 0
      self.cutn: float = 1
      self.cut_pow: float = 0.0001
      self.init_mse_scale: float = 0
      self.init_mse_image: str = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"
      self.blue_scale: float = 0
      self.gradient_wrt = "'x0_pred'"  # x, x0_pred
      self.gradient_add_to = "'both'"  # cond, uncond, both
      self.decode_method = "'linear'"  # autoencoder, linear
      self.grad_threshold_type = "'dynamic'"  # dynamic, static, mean, schedule
      self.clamp_grad_threshold: float = 0.2
      self.clamp_start = "0.2"
      self.clamp_stop = "0.01"
      self.grad_inject_timing = "list(range(1,10))"
      self.cond_uncond_sync: bool = True
class Root(ChangeRecorder):
  def __init__(self):
      super().__init__()
      self.models_path: str = "models"
      self.configs_path: str = "configs"
      self.output_path: str = "outputs"
      self.mount_google_drive: bool = True
      self.models_path_gdrive: str = "/content/drive/MyDrive/AI/models"
      self.output_path_gdrive: str = "/content/drive/MyDrive/AI/StableDiffusion"
      self.map_location = "cuda"  # cpu, cuda
      self.model_config = "v1-inference.yaml"  # custom, v2-inference.yaml, v2-inference-v.yaml, v1-inference.yaml
      self.model_checkpoint = "Protogen_V2.2.ckpt"  # custom, v2-1_768-ema-pruned.ckpt, v2-1_512-ema-pruned.ckpt, 768-v-ema.ckpt, 512-base-ema.ckpt, Protogen_V2.2.ckpt, v1-5-pruned.ckpt, v1-5-pruned-emaonly.ckpt, sd-v1-4-full-ema.ckpt, sd-v1-4.ckpt, sd-v1-3-full-ema.ckpt, sd-v1-3.ckpt, sd-v1-2-full-ema.ckpt, sd-v1-2.ckpt, sd-v1-1-full-ema.ckpt, sd-v1-1.ckpt, robo-diffusion-v1.ckpt, wd-v1-3-float16.ckpt
      self.custom_config_path: str = ""
      self.custom_checkpoint_path: str = ""
class DeforumAnimArgs(ChangeRecorder):
  def __init__(self):
      super().__init__()
      self.animation_mode: str = "'2D'"  # None, 2D, 3D, Video Input, Interpolation
      self.max_frames: float = 270
      self.border: str = "'replicate'"  # wrap, replicate
      self.angle: str = "0:(0)"
      self.zoom: str = "0: (1.22), 1: (1.06), 2: (1.04), 3: (1.18), 4: (1.06), 5: (1.04), 6: (1.16), 7: (1.05), 8: (1.04), 9: (1.17), 10: (1.15), 11: (1.04), 12: (1.18), 13: (1.05), 14: (1.04), 15: (1.14), 16: (1.06), 17: (1.04), 18: (1.15), 19: (1.05), 20: (1.04), 21: (1.16), 22: (1.16), 23: (1.04), 24: (1.18), 25: (1.05), 26: (1.04), 27: (1.15), 28: (1.06), 29: (1.04), 30: (1.18), 31: (1.05), 32: (1.04), 33: (1.16), 34: (1.14), 35: (1.04), 36: (1.19), 37: (1.05), 38: (1.04), 39: (1.18), 40: (1.06), 41: (1.04), 42: (1.20), 43: (1.05), 44: (1.04), 45: (1.15), 46: (1.13), 47: (1.04), 48: (1.16), 49: (1.05), 50: (1.04), 51: (1.18), 52: (1.06), 53: (1.04), 54: (1.20), 55: (1.05), 56: (1.04), 57: (1.15), 58: (1.13), 59: (1.04), 60: (1.19), 61: (1.06), 62: (1.04), 63: (1.17), 64: (1.06), 65: (1.04), 66: (1.18), 67: (1.05), 68: (1.04), 69: (1.15), 70: (1.17), 71: (1.04), 72: (1.14), 73: (1.05), 74: (1.04), 75: (1.19), 76: (1.06), 77: (1.04), 78: (1.21), 79: (1.05), 80: (1.04), 81: (1.16), 82: (1.14), 83: (1.04), 84: (1.15), 85: (1.06), 86: (1.04), 87: (1.19), 88: (1.06), 89: (1.04), 90: (1.20), 91: (1.05), 92: (1.04), 93: (1.15), 94: (1.13), 95: (1.04), 96: (1.22), 97: (1.05), 98: (1.04), 99: (1.21), 100: (1.05), 101: (1.04), 102: (1.20), 103: (1.05), 104: (1.04), 105: (1.14), 106: (1.14), 107: (1.04), 108: (1.18), 109: (1.05), 110: (1.04), 111: (1.17), 112: (1.05), 113: (1.04), 114: (1.23), 115: (1.05), 116: (1.04), 117: (1.19), 118: (1.14), 119: (1.04), 120: (1.17), 121: (1.05), 122: (1.04), 123: (1.17), 124: (1.05), 125: (1.04), 126: (1.19), 127: (1.04), 128: (1.04), 129: (1.17), 130: (1.13), 131: (1.04), 132: (1.16), 133: (1.05), 134: (1.04), 135: (1.17), 136: (1.05), 137: (1.04), 138: (1.20), 139: (1.04), 140: (1.04), 141: (1.14), 142: (1.13), 143: (1.04), 144: (1.20), 145: (1.05), 146: (1.04), 147: (1.16), 148: (1.05), 149: (1.04), 150: (1.19), 151: (1.05), 152: (1.04), 153: (1.14), 154: (1.11), 155: (1.04), 156: (1.19), 157: (1.05), 158: (1.04), 159: (1.19), 160: (1.05), 161: (1.04), 162: (1.16), 163: (1.04), 164: (1.04), 165: (1.15), 166: (1.13), 167: (1.04), 168: (1.04), 169: (1.04), 170: (1.04), 171: (1.04), 172: (1.04), 173: (1.04), 174: (1.54), 175: (1.24), 176: (1.05), 177: (1.43), 178: (1.21), 179: (1.06), 180: (1.37), 181: (1.25), 182: (1.05), 183: (1.85), 184: (2.02), 185: (1.06), 186: (1.55), 187: (1.25), 188: (1.06), 189: (1.44), 190: (1.19), 191: (1.06), 192: (1.51), 193: (1.20), 194: (1.06), 195: (1.50), 196: (1.22), 197: (1.50), 198: (1.49), 199: (1.21), 200: (1.06), 201: (1.44), 202: (1.22), 203: (1.06), 204: (1.47), 205: (1.20), 206: (1.05), 207: (1.75), 208: (1.82), 209: (1.06), 210: (1.52), 211: (1.23), 212: (1.06), 213: (1.52), 214: (1.21), 215: (1.06), 216: (1.46), 217: (1.26), 218: (1.06), 219: (1.47), 220: (1.21), 221: (1.49), 222: (1.45), 223: (1.21), 224: (1.06), 225: (1.52), 226: (1.21), 227: (1.06), 228: (1.43), 229: (1.24), 230: (1.06), 231: (1.83), 232: (1.80), 233: (1.05), 234: (1.55), 235: (1.20), 236: (1.05), 237: (1.41), 238: (1.23), 239: (1.06), 240: (1.42), 241: (1.22), 242: (1.06), 243: (1.50), 244: (1.24), 245: (1.56), 246: (1.50), 247: (1.20), 248: (1.05), 249: (1.42), 250: (1.21), 251: (1.05), 252: (1.39), 253: (1.22), 254: (1.06), 255: (1.74), 256: (2.04), 257: (1.06), 258: (1.45), 259: (1.19), 260: (1.05), 261: (1.48), 262: (1.22), 263: (1.06), 264: (1.46), 265: (1.14), 266: (1.04), 267: (1.12), 268: (1.04), 269: (1.04), 270: (1.04)"
      self.translation_x: str = "0:(0)"
      self.translation_y: str = "0:(3)"
      self.translation_z: str = "0:(0)"
      self.rotation_3d_x: str = "0:(0)"
      self.rotation_3d_y: str = "0:(0)"
      self.rotation_3d_z: str = "0:(0)"
      self.flip_2d_perspective: bool = False
      self.perspective_flip_theta: str = "0:(0)"
      self.perspective_flip_phi: str = "0:(0)"
      self.perspective_flip_gamma: str = "0:(0)"
      self.perspective_flip_fv: str = "0:(53)"
      self.noise_schedule: str = "0: (0.02)"
      self.strength_schedule: str = "0: (0.70)"
      self.contrast_schedule: str = "0: (1.0)"
      self.hybrid_video_comp_alpha_schedule: str = "0:(1)"
      self.hybrid_video_comp_mask_blend_alpha_schedule: str = "0:(0.5)"
      self.hybrid_video_comp_mask_contrast_schedule: str = "0:(1)"
      self.hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule: str = "0:(100)"
      self.hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule: str = "0:(0)"
      self.kernel_schedule: str = "0: (5)"
      self.sigma_schedule: str = "0: (1.0)"
      self.amount_schedule: str = "0: (0.2)"
      self.threshold_schedule: str = "0: (0.0)"
      self.color_coherence: str = "'Match Frame 0 LAB'"  # None, Match Frame 0 HSV, Match Frame 0 LAB, Match Frame 0 RGB, Video Input
      self.color_coherence_video_every_N_frames: int = 1
      self.diffusion_cadence: str = "'1'"  # 1, 2, 3, 4, 5, 6, 7, 8
      self.use_depth_warping: bool = True
      self.midas_weight: float = 0.3
      self.fov: float = 40
      self.padding_mode: str = "'border'"  # border, reflection, zeros
      self.sampling_mode: str = "'bicubic'"  # bicubic, bilinear, nearest
      self.save_depth_maps: bool = False
      self.video_init_path: str = "'/content/video_in.mp4'"
      self.extract_nth_frame: float = 1
      self.overwrite_extracted_frames: bool = True
      self.use_mask_video: bool = False
      self.video_mask_path: str = "'/content/video_in.mp4'"
      self.hybrid_video_generate_inputframes: bool = False
      self.hybrid_video_use_first_frame_as_init_image: bool = True
      self.hybrid_video_motion = "None"  # None, Optical Flow, Perspective, Affine
      self.hybrid_video_flow_method = "Farneback"  # Farneback, DenseRLOF, SF
      self.hybrid_video_composite: bool = False
      self.hybrid_video_comp_mask_type = "None"  # None, Depth, Video Depth, Blend, Difference
      self.hybrid_video_comp_mask_inverse: bool = False
      self.hybrid_video_comp_mask_equalize = "None"  # None, Before, After, Both
      self.hybrid_video_comp_mask_auto_contrast: bool = False
      self.hybrid_video_comp_save_extra_frames: bool = False
      self.hybrid_video_use_video_as_mse_image: bool = False
      self.interpolate_key_frames: bool = False
      self.interpolate_x_frames: float = 4
      self.resume_from_timestring: bool = False
      self.resume_timestring: str = "20220829210106"
