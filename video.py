# %% [markdown]
# # **Deforum Stable Diffusion on FILM v0.8**
# 
# A blending of Deforum and Google FILM (for ultra high quality frame interpolation) to create ultra-smooth deforum videos. 
# 
# [Stable Diffusion](https://github.com/CompVis/stable-diffusion) by Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer and the [Stability.ai](https://stability.ai/) Team. [K Diffusion](https://github.com/crowsonkb/k-diffusion) by [Katherine Crowson](https://twitter.com/RiversHaveWings). Notebook by [deforum](https://discord.gg/upmXXsrwZc) Google FILM (frame interpolation) integration by [boxabirds](https://github.com/boxabirds)
# 

# %%
#@markdown **NVIDIA GPU**
import torch
import subprocess, os, sys

if torch.cuda.is_available():
    sub_p_res = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(f"{sub_p_res[:-1]}")
    has_cuda = True
else:
    print("No Nvidia GPU software or hardware found")

# %% [markdown]
# # Setup

# %%
#@markdown **Environment Setup**
import subprocess, time, gc, os, sys

def setup_environment():
    start_time = time.time()
    print_subprocess = False
    use_xformers_for_colab = True
    try:
        ipy = get_ipython()
    except:
        ipy = 'could not get_ipython'
    if 'google.colab' in str(ipy):
        print("..setting up environment")

        # weird hack
        #import torch
        
        all_process = [
            ['pip', 'install', 'omegaconf', 'einops==0.4.1', 'pytorch-lightning==1.7.7', 'torchmetrics', 'transformers', 'safetensors', 'kornia'],
            ['git', 'clone', 'https://github.com/boxabirds/deforum-stable-diffusion'],
            ['pip', 'install', 'accelerate', 'ftfy', 'jsonmerge', 'matplotlib', 'resize-right', 'timm', 'torchdiffeq','scikit-learn','torchsde','open-clip-torch','numpngw'],
        ]
        for process in all_process:
            running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
            if print_subprocess:
                print(running)
        with open('deforum-stable-diffusion/src/k_diffusion/__init__.py', 'w') as f:
            f.write('')
        sys.path.extend([
            'deforum-stable-diffusion/',
            'deforum-stable-diffusion/src',
        ])
        if use_xformers_for_colab:

            print("..installing triton and xformers")

            all_process = [['pip', 'install', 'triton', 'xformers']]
            for process in all_process:
                running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
                if print_subprocess:
                    print(running)
    else:
        sys.path.extend([
            'src'
        ])
    end_time = time.time()
    print(f"..environment set up in {end_time-start_time:.0f} seconds")
    return

setup_environment()


import random
import clip
from IPython import display

from helpers.settings import load_args
from helpers.render import do_render
from helpers.model_load import make_linear_decode, load_model, get_model_output_paths
from helpers.aesthetics import load_aesthetics_model
from helpers.video import frames2vid

# %%
#@markdown **Path Setup**
from types import SimpleNamespace
from helpers.save_images import get_output_folder

def Root():
    models_path = "models" #@param {type:"string"}
    configs_path = "configs" #@param {type:"string"}
    output_path = "outputs" #@param {type:"string"}
    mount_google_drive = True #@param {type:"boolean"}
    models_path_gdrive = "/content/drive/MyDrive/AI/models" #@param {type:"string"}
    output_path_gdrive = "/content/drive/MyDrive/AI/StableDiffusion" #@param {type:"string"}

    #@markdown **Model Setup**
    map_location = "cuda" #@param ["cpu", "cuda"]
    print( f"map_location: {map_location}" )
    model_config = "v1-inference.yaml" #@param ["custom","v2-inference.yaml","v2-inference-v.yaml","v1-inference.yaml"]
    model_checkpoint =  "Protogen_V2.2.ckpt" #@param ["custom","v2-1_768-ema-pruned.ckpt","v2-1_512-ema-pruned.ckpt","768-v-ema.ckpt","512-base-ema.ckpt","Protogen_V2.2.ckpt","v1-5-pruned.ckpt","v1-5-pruned-emaonly.ckpt","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt", "robo-diffusion-v1.ckpt","wd-v1-3-float16.ckpt"]
    custom_config_path = "" #@param {type:"string"}
    custom_checkpoint_path = "" #@param {type:"string"}
    return locals()

root = Root()
root = SimpleNamespace(**root)

root.models_path, root.output_path = get_model_output_paths(root)

# %%
root.model, root.device = load_model(root, load_on_run_all=True, check_sha256=True, map_location=root.map_location)

# %% [markdown]
# # Settings

# %%
def DeforumAnimArgs():

    #@markdown ####**Animation:**
    animation_mode = '2D' #@param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
    max_frames = 1040 #@param {type:"number"}
    border = 'replicate' #@param ['wrap', 'replicate'] {type:'string'}

    #@markdown ####**Motion Parameters:**
    angle = "0:(0)"#@param {type:"string"}
    zoom = "0: (1.01), 5: (1.40), 6: (1.20), 7: (1.10), 8: (1.40), 9: (1.20), 10: (1.10), 11: (1.40), 12: (1.20), 13: (1.10), 14: (1.40), 15: (1.20), 16: (1.10), 17: (1.40), 18: (1.20), 19: (1.10), 20: (1.40), 21: (1.20), 22: (1.40), 23: (1.40), 24: (1.20), 25: (1.10), 26: (1.40), 27: (1.20), 28: (1.10), 29: (1.40), 30: (1.20), 31: (1.10), 32: (1.40), 33: (1.20), 34: (1.10), 35: (1.01), 38: (1.40), 39: (1.20), 40: (1.10), 41: (1.01), 47: (1.40), 48: (1.20), 49: (1.10), 50: (1.01), 53: (1.40), 54: (1.20), 55: (1.10), 56: (1.40), 57: (1.20), 58: (1.10), 59: (1.40), 60: (1.20), 61: (1.10), 62: (1.40), 63: (1.20), 64: (1.10), 65: (1.40), 66: (1.20), 67: (1.10), 68: (1.40), 69: (1.20), 70: (1.10), 71: (1.40), 72: (1.20), 73: (1.10), 74: (1.40), 75: (1.20), 76: (1.10), 77: (1.40), 78: (1.20), 79: (1.10), 80: (1.01), 95: (1.40), 96: (1.20), 97: (1.10), 98: (1.40), 99: (1.20), 100: (1.10), 101: (1.40), 102: (1.20), 103: (1.10), 104: (1.40), 105: (1.20), 106: (1.40), 107: (1.40), 108: (1.40), 109: (1.20), 110: (1.40), 111: (1.20), 112: (1.10), 113: (1.40), 114: (1.20), 115: (1.10), 116: (1.40), 117: (1.20), 118: (1.10), 119: (1.40), 120: (1.20), 121: (1.10), 122: (1.40), 123: (1.20), 124: (1.10), 125: (1.40), 126: (1.20), 127: (1.10), 128: (1.01), 143: (1.40), 144: (1.20), 145: (1.10), 146: (1.40), 147: (1.20), 148: (1.10), 149: (1.40), 150: (1.20), 151: (1.10), 152: (1.40), 153: (1.20), 154: (1.40), 155: (1.40), 156: (1.20), 157: (1.10), 158: (1.40), 159: (1.20), 160: (1.10), 161: (1.40), 162: (1.20), 163: (1.10), 164: (1.40), 165: (1.20), 166: (1.10), 167: (1.40), 168: (1.20), 169: (1.10), 170: (1.40), 171: (1.20), 172: (1.10), 173: (1.01), 183: (1.40), 184: (1.20), 185: (1.10), 186: (1.01), 188: (1.40), 189: (1.20), 190: (1.10), 191: (1.40), 192: (1.40), 193: (1.20), 194: (1.40), 195: (1.20), 196: (1.10), 197: (1.40), 198: (1.20), 199: (1.10), 200: (1.40), 201: (1.20), 202: (1.10), 203: (1.40), 204: (1.20), 205: (1.10), 206: (1.01), 212: (1.40), 213: (1.20), 214: (1.40), 215: (1.40), 216: (1.20), 217: (1.40), 218: (1.40), 219: (1.20), 220: (1.40), 221: (1.40), 222: (1.20), 223: (1.40), 224: (1.40), 225: (1.20), 226: (1.40), 227: (1.20), 228: (1.10), 229: (1.40), 230: (1.20), 231: (1.10), 232: (1.40), 233: (1.40), 234: (1.20), 235: (1.40), 236: (1.40), 237: (1.20), 238: (1.10), 239: (1.40), 240: (1.20), 241: (1.10), 242: (1.40), 243: (1.20), 244: (1.40), 245: (1.40), 246: (1.20), 247: (1.10), 248: (1.40), 249: (1.20), 250: (1.10), 251: (1.40), 252: (1.20), 253: (1.10), 254: (1.40), 255: (1.20), 256: (1.40), 257: (1.40), 258: (1.20), 259: (1.40), 260: (1.40), 261: (1.20), 262: (1.40), 263: (1.40), 264: (1.20), 265: (1.40), 266: (1.40), 267: (1.20), 268: (1.40), 269: (1.20), 270: (1.10), 271: (1.40), 272: (1.20), 273: (1.10), 274: (1.40), 275: (1.20), 276: (1.10), 277: (1.40), 278: (1.40), 279: (1.20), 280: (1.40), 281: (1.40), 282: (1.20), 283: (1.40), 284: (1.40), 285: (1.20), 286: (1.10), 287: (1.40), 288: (1.20), 289: (1.10), 290: (1.40), 291: (1.20), 292: (1.40), 293: (1.40), 294: (1.20), 295: (1.10), 296: (1.40), 297: (1.20), 298: (1.10), 299: (1.40), 300: (1.20), 301: (1.10), 302: (1.40), 303: (1.20), 304: (1.40), 305: (1.40), 306: (1.20), 307: (1.40), 308: (1.20), 309: (1.10), 310: (1.01), 313: (1.40), 314: (1.20), 315: (1.10), 316: (1.01), 323: (1.40), 324: (1.20), 325: (1.10), 326: (1.40), 327: (1.20), 328: (1.10), 329: (1.40), 330: (1.20), 331: (1.10), 332: (1.01), 335: (1.40), 336: (1.20), 337: (1.10), 338: (1.40), 339: (1.20), 340: (1.10), 341: (1.40), 342: (1.20), 343: (1.10), 344: (1.40), 345: (1.20), 346: (1.40), 347: (1.40), 348: (1.20), 349: (1.10), 350: (1.01), 365: (1.40), 366: (1.20), 367: (1.10), 368: (1.40), 369: (1.20), 370: (1.10), 371: (1.40), 372: (1.20), 373: (1.10), 374: (1.40), 375: (1.20), 376: (1.10), 377: (1.40), 378: (1.20), 379: (1.10), 380: (1.40), 381: (1.20), 382: (1.10), 383: (1.40), 384: (1.20), 385: (1.10), 386: (1.40), 387: (1.20), 388: (1.10), 389: (1.40), 390: (1.20), 391: (1.10), 392: (1.40), 393: (1.20), 394: (1.40), 395: (1.40), 396: (1.20), 397: (1.40), 398: (1.20), 399: (1.10), 400: (1.01), 413: (1.40), 414: (1.20), 415: (1.10), 416: (1.40), 417: (1.20), 418: (1.10), 419: (1.40), 420: (1.20), 421: (1.10), 422: (1.40), 423: (1.20), 424: (1.10), 425: (1.40), 426: (1.20), 427: (1.10), 428: (1.40), 429: (1.20), 430: (1.10), 431: (1.40), 432: (1.20), 433: (1.10), 434: (1.40), 435: (1.20), 436: (1.40), 437: (1.40), 438: (1.20), 439: (1.40), 440: (1.40), 441: (1.20), 442: (1.40), 443: (1.20), 444: (1.10), 445: (1.40), 446: (1.20), 447: (1.10), 448: (1.01), 449: (1.40), 450: (1.20), 451: (1.10), 452: (1.01), 457: (1.40), 458: (1.20), 459: (1.10), 460: (1.40), 461: (1.40), 462: (1.20), 463: (1.10), 464: (1.40), 465: (1.40), 466: (1.20), 467: (1.40), 468: (1.20), 469: (1.10), 470: (1.40), 471: (1.20), 472: (1.10), 473: (1.40), 474: (1.20), 475: (1.10), 476: (1.40), 477: (1.40), 478: (1.40), 479: (1.40), 480: (1.20), 481: (1.40), 482: (1.40), 483: (1.20), 484: (1.40), 485: (1.40), 486: (1.20), 487: (1.40), 488: (1.40), 489: (1.40), 490: (1.40), 491: (1.40), 492: (1.20), 493: (1.40), 494: (1.40), 495: (1.20), 496: (1.40), 497: (1.20), 498: (1.10), 499: (1.40), 500: (1.20), 501: (1.10), 502: (1.40), 503: (1.20), 504: (1.10), 505: (1.01), 506: (1.40), 507: (1.20), 508: (1.40), 509: (1.40), 510: (1.20), 511: (1.10), 512: (1.40), 513: (1.40), 514: (1.20), 515: (1.40), 516: (1.20), 517: (1.10), 518: (1.40), 519: (1.20), 520: (1.10), 521: (1.40), 522: (1.20), 523: (1.40), 524: (1.40), 525: (1.40), 526: (1.40), 527: (1.40), 528: (1.20), 529: (1.40), 530: (1.40), 531: (1.20), 532: (1.40), 533: (1.40), 534: (1.20), 535: (1.40), 536: (1.40), 537: (1.20), 538: (1.40), 539: (1.40), 540: (1.20), 541: (1.10), 542: (1.01), 544: (1.40), 545: (1.20), 546: (1.10), 547: (1.40), 548: (1.20), 549: (1.10), 550: (1.40), 551: (1.40), 552: (1.20), 553: (1.10), 554: (1.01), 559: (1.40), 560: (1.40), 561: (1.20), 562: (1.10), 563: (1.40), 564: (1.20), 565: (1.10), 566: (1.40), 567: (1.40), 568: (1.20), 569: (1.40), 570: (1.20), 571: (1.10), 572: (1.40), 573: (1.20), 574: (1.40), 575: (1.40), 576: (1.20), 577: (1.40), 578: (1.40), 579: (1.40), 580: (1.40), 581: (1.40), 582: (1.20), 583: (1.40), 584: (1.40), 585: (1.20), 586: (1.40), 587: (1.20), 588: (1.10), 589: (1.40), 590: (1.40), 591: (1.20), 592: (1.40), 593: (1.20), 594: (1.10), 595: (1.40), 596: (1.40), 597: (1.20), 598: (1.40), 599: (1.40), 600: (1.20), 601: (1.40), 602: (1.40), 603: (1.40), 604: (1.20), 605: (1.40), 606: (1.20), 607: (1.10), 608: (1.40), 609: (1.20), 610: (1.10), 611: (1.40), 612: (1.20), 613: (1.10), 614: (1.40), 615: (1.20), 616: (1.10), 617: (1.40), 618: (1.20), 619: (1.40), 620: (1.40), 621: (1.20), 622: (1.40), 623: (1.40), 624: (1.20), 625: (1.40), 626: (1.40), 627: (1.40), 628: (1.40), 629: (1.40), 630: (1.20), 631: (1.40), 632: (1.40), 633: (1.20), 634: (1.40), 635: (1.20), 636: (1.10), 637: (1.40), 638: (1.40), 639: (1.20), 640: (1.40), 641: (1.40), 642: (1.20), 643: (1.40), 644: (1.40), 645: (1.20), 646: (1.40), 647: (1.40), 648: (1.20), 649: (1.40), 650: (1.40), 651: (1.40), 652: (1.20), 653: (1.40), 654: (1.20), 655: (1.10), 656: (1.01), 662: (1.40), 663: (1.20), 664: (1.10), 665: (1.40), 666: (1.20), 667: (1.10), 668: (1.01), 683: (1.40), 684: (1.20), 685: (1.10), 686: (1.40), 687: (1.20), 688: (1.10), 689: (1.40), 690: (1.20), 691: (1.10), 692: (1.40), 693: (1.20), 694: (1.10), 695: (1.40), 696: (1.20), 697: (1.10), 698: (1.40), 699: (1.20), 700: (1.10), 701: (1.40), 702: (1.20), 703: (1.10), 704: (1.40), 705: (1.20), 706: (1.40), 707: (1.20), 708: (1.10), 709: (1.01), 710: (1.40), 711: (1.20), 712: (1.10), 713: (1.01), 718: (1.40), 719: (1.20), 720: (1.10), 721: (1.01), 725: (1.40), 726: (1.20), 727: (1.10), 728: (1.40), 729: (1.20), 730: (1.10), 731: (1.40), 732: (1.20), 733: (1.10), 734: (1.40), 735: (1.20), 736: (1.10), 737: (1.40), 738: (1.20), 739: (1.10), 740: (1.40), 741: (1.20), 742: (1.10), 743: (1.40), 744: (1.20), 745: (1.10), 746: (1.40), 747: (1.20), 748: (1.10), 749: (1.40), 750: (1.20), 751: (1.40), 752: (1.20), 753: (1.10), 754: (1.01), 773: (1.40), 774: (1.20), 775: (1.10), 776: (1.40), 777: (1.20), 778: (1.10), 779: (1.40), 780: (1.20), 781: (1.10), 782: (1.40), 783: (1.20), 784: (1.10), 785: (1.40), 786: (1.20), 787: (1.10), 788: (1.40), 789: (1.20), 790: (1.10), 791: (1.40), 792: (1.20), 793: (1.10), 794: (1.40), 795: (1.20), 796: (1.10), 797: (1.40), 798: (1.20), 799: (1.10), 800: (1.01), 814: (1.40), 815: (1.20), 816: (1.10), 817: (1.01), 865: (1.40), 866: (1.40), 867: (1.20), 868: (1.40), 869: (1.40), 870: (1.20), 871: (1.40), 872: (1.40), 873: (1.40), 874: (1.40), 875: (1.40), 876: (1.20), 877: (1.40), 878: (1.40), 879: (1.20), 880: (1.10), 881: (1.40), 882: (1.20), 883: (1.10), 884: (1.40), 885: (1.40), 886: (1.40), 887: (1.40), 888: (1.20), 889: (1.40), 890: (1.40), 891: (1.20), 892: (1.40), 893: (1.40), 894: (1.20), 895: (1.40), 896: (1.40), 897: (1.40), 898: (1.40), 899: (1.40), 900: (1.20), 901: (1.40), 902: (1.40), 903: (1.20), 904: (1.40), 905: (1.20), 906: (1.10), 907: (1.40), 908: (1.40), 909: (1.20), 910: (1.40), 911: (1.40), 912: (1.20), 913: (1.40), 914: (1.40), 915: (1.20), 916: (1.40), 917: (1.40), 918: (1.20), 919: (1.40), 920: (1.40), 921: (1.40), 922: (1.20), 923: (1.40), 924: (1.20), 925: (1.10), 926: (1.40), 927: (1.20), 928: (1.10), 929: (1.40), 930: (1.20), 931: (1.10), 932: (1.40), 933: (1.40), 934: (1.40), 935: (1.40), 936: (1.20), 937: (1.40), 938: (1.40), 939: (1.20), 940: (1.40), 941: (1.40), 942: (1.20), 943: (1.40), 944: (1.40), 945: (1.20), 946: (1.40), 947: (1.40), 948: (1.20), 949: (1.40), 950: (1.20), 951: (1.10), 952: (1.40), 953: (1.20), 954: (1.10), 955: (1.40), 956: (1.40), 957: (1.20), 958: (1.40), 959: (1.40), 960: (1.20), 961: (1.10), 962: (1.01)"#@param {type:"string"}
    translation_x = "0:(0)"#@param {type:"string"}
    translation_y = "0:(3)"#@param {type:"string"}
    translation_z = "0:(0)"#@param {type:"string"}
    rotation_3d_x = "0:(0)"#@param {type:"string"}
    rotation_3d_y = "0:(0)"#@param {type:"string"}
    rotation_3d_z = "0:(0)"#@param {type:"string"}
    flip_2d_perspective = False #@param {type:"boolean"}
    perspective_flip_theta = "0:(0)"#@param {type:"string"}
    perspective_flip_phi = "0:(0)"#@param {type:"string"}
    perspective_flip_gamma = "0:(0)"#@param {type:"string"}
    perspective_flip_fv = "0:(53)"#@param {type:"string"}
    noise_schedule = "0: (0.02)"#@param {type:"string"}
    strength_schedule = "0: (0.65)"#@param {type:"string"}
    contrast_schedule = "0: (1.0)"#@param {type:"string"}
    hybrid_video_comp_alpha_schedule = "0:(1)" #@param {type:"string"}
    hybrid_video_comp_mask_blend_alpha_schedule = "0:(0.5)" #@param {type:"string"}
    hybrid_video_comp_mask_contrast_schedule = "0:(1)" #@param {type:"string"}
    hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule =  "0:(100)" #@param {type:"string"}
    hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule =  "0:(0)" #@param {type:"string"}

    #@markdown ####**Unsharp mask (anti-blur) Parameters:**
    kernel_schedule = "0: (5)"#@param {type:"string"}
    sigma_schedule = "0: (1.0)"#@param {type:"string"}
    amount_schedule = "0: (0.2)"#@param {type:"string"}
    threshold_schedule = "0: (0.0)"#@param {type:"string"}

    #@markdown ####**Coherence:**
    color_coherence = 'Match Frame 0 LAB' #@param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB', 'Video Input'] {type:'string'}
    color_coherence_video_every_N_frames = 1 #@param {type:"integer"}
    diffusion_cadence = '1' #@param ['1','2','3','4','5','6','7','8'] {type:'string'}

    #@markdown ####**3D Depth Warping:**
    use_depth_warping = True #@param {type:"boolean"}
    midas_weight = 0.3#@param {type:"number"}
    near_plane = 200
    far_plane = 10000
    fov = 40#@param {type:"number"}
    padding_mode = 'border'#@param ['border', 'reflection', 'zeros'] {type:'string'}
    sampling_mode = 'bicubic'#@param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
    save_depth_maps = False #@param {type:"boolean"}

    #@markdown ####**Video Input:**
    video_init_path ='/content/video_in.mp4'#@param {type:"string"}
    extract_nth_frame = 1#@param {type:"number"}
    overwrite_extracted_frames = True #@param {type:"boolean"}
    use_mask_video = False #@param {type:"boolean"}
    video_mask_path ='/content/video_in.mp4'#@param {type:"string"}

    #@markdown ####**Hybrid Video for 2D/3D Animation Mode:**
    hybrid_video_generate_inputframes = False #@param {type:"boolean"}
    hybrid_video_use_first_frame_as_init_image = True #@param {type:"boolean"}
    hybrid_video_motion = "None" #@param ['None','Optical Flow','Perspective','Affine']
    hybrid_video_flow_method = "Farneback" #@param ['Farneback','DenseRLOF','SF']
    hybrid_video_composite = False #@param {type:"boolean"}
    hybrid_video_comp_mask_type = "None" #@param ['None', 'Depth', 'Video Depth', 'Blend', 'Difference']
    hybrid_video_comp_mask_inverse = False #@param {type:"boolean"}
    hybrid_video_comp_mask_equalize = "None" #@param  ['None','Before','After','Both']
    hybrid_video_comp_mask_auto_contrast = False #@param {type:"boolean"}
    hybrid_video_comp_save_extra_frames = False #@param {type:"boolean"}
    hybrid_video_use_video_as_mse_image = False #@param {type:"boolean"}

    #@markdown ####**Interpolation:**
    interpolate_key_frames = False #@param {type:"boolean"}
    interpolate_x_frames = 4 #@param {type:"number"}
    
    #@markdown ####**Resume Animation:**
    resume_from_timestring = False #@param {type:"boolean"}
    resume_timestring = "20220829210106" #@param {type:"string"}

    return locals()

# %%
prompts = [
    "a beautiful lake by Asher Brown Durand, trending on Artstation", # the first prompt I want
    "a beautiful portrait of a woman by Artgerm, trending on Artstation", # the second prompt I want
    #"this prompt I don't want it I commented it out",
    #"a nousr robot, trending on Artstation", # use "nousr robot" with the robot diffusion model (see model_checkpoint setting)
    #"touhou 1girl komeiji_koishi portrait, green hair", # waifu diffusion prompts can use danbooru tag groups (see model_checkpoint)
    #"this prompt has weights if prompt weighting enabled:2 can also do negative:-2", # (see prompt_weighting)
]

# raw_animation_prompts = {
# 0: "People wearing coats by the seaside in winter with dark grey skies",
# 32: "Driving in a car along a motorway with a beach and waves in the distance",
# 64: "Rolling green landscapes set against a grey sky, watercolor",
# 95: "3 adults walking along a wintry pebble beach bracing against strong winds",
# 127: "Close up of 3 adults on a wintry pebble beach looking happy, healthy and revitalised",
# 160: "3 adults walking along a wintry pebble beach stepping over stones",
# 192: "3 adults on wintry pebble beach sitting on a picnic mat that is fluttering in the wind",
# 233: "Office scene with people pretending to be nice to each other",
# 247: "People in office with tables, chairs, and sofas of different colours next to a table with coffee and tea cups",
# 264: "Close-up of two people in an office whispering to each other conspiratorially next to a water cooler",
# 279: "Office team cheering and slapping each other on each other's backs",
# 297: "Miserable people sitting on an underground railway that is falling apart",
# 311: "Sad person in their home holding bills, surrounded by a mountain of other bills",
# 329: "Person thinking philosophically, looking up into the sky",
# 344: "Person thinking philosophically, throwing their hands up into the sky in frustration",
# }
# style_extras = "watercolor, cosy, trending on Artstation" #@param {type:"string"}

raw_animation_prompts = {
0: "Moody winter seaside landscape with rolling green hills and grey skies.",
60: "A car driving amongst green hills towards waves at a beach in the distance.",
84: "Vivid green rolling hills contrasted with grey skies, a small figure standing on the shoreline, facing the horizon with arms outstretched in defiance.",
108: "A person in a winter coat standing at the edge of a beach, looking out to sea with the wind blowing their hair back, a determined expression on their face.",
132: "3 people in coats, bare feet on stones walking across the beach.",
156: "Person standing in the rain on a beach, looking out to waves crashing against the shore.",
180: "3 people sitting on a picnic blanket on a stony beach flapping in the wind",
211: "An office environment with people sitting at desks, talking and laughing.",
222: "Brightly coloured chairs, sofas and carpet in a bright office, with a group of people laughing and drinking free coffee and tea together.",
234: "People in an office gathered in a circle around a water cooler, laughing and talking.",
244: "A group of office workers gathered around a table, cheering and clapping in a team building exercise",
259: "A bus full of people held together with duct tape and patches in a busy city.",
270: "A family of four at home looking at a pile of bills with electric company logos.",
283: "A person standing on a beach in the rain with the waves crashing against the shore in the background.",
294: "Person standing alone on a beach, looking out to sea with a contemplative expression on their face.",
306: "Two people standing side by side ankle deep in the water surrounded by splashing waves, looking out to a vast horizon of stormy gray clouds, with a suggestion of a shimmering rainbow in the distance.",
329: "Two people swimmming in the ocean, smiles on their faces as the cold salty mist of the waves hit their skin.",
354: "Vividly-colored beachgoers emerging from the waves with flushed cheeks and wide grins.",
378: "Wet Wind-swept figures emerging from the ocean after a swim, skin covered in pale goosebumps.",
402: "3 people eating fish & chips on the beach with newspaper flapping around",
425: "Seagulls dive bombing a picnic table with a family eating defiantly.",
457: "An office environment with people sitting at desks, talking and laughing.",
468: "Rainbow-hued furniture in an office, with cups of tea and coffee and kettle and coffee maker", 
480: "Two people huddled together around a water cooler in the office, smiling and talking.",
490: "A group of office workers gathered around a table, looking angry with knives and bats attacking each other",
505: "Inside an underground train full of people, held together with duct tape and patches.",
516: "Image of person at home, staring at a huge room full of eletricity bills",
529: "A person standing in a winter landscape, with a hooded coat and hands tucked into pockets, looking out at the horizon with a pensive expression.",
540: "Person standing on a beach, looking out at the endless horizon of the sea, fists clenched in a determined pose.",
558: "People standing at a beach, the horizon glowing with vibrant oranges and pinks as the sun sets, a lone seagull flying overhead.",
570: "A group of people standing on a beach with their arms raised in the air, looking out towards the horizon with a look of determination and joy on their faces.",
587: "A couple embracing against a backdrop of rolling waves crashing against a rocky shoreline.",
606: "A person standing in the ocean, arms outstretched and face to the sky, with a rainbow overhead.",
616: "Vibrant sunset illuminating a group of people jumping into the sea, with arms spread wide.",
635: "A group of people jumping into a wild and turbulent ocean, joyfully embracing the power of the waves.",
660: "Rainbow-coloured umbrellas against a backdrop of grey clouds, with people emerging from the waves below, embracing the feeling of being alive.",
684: "Person standing in the rain, arms outstretched, eyes closed, embracing the feeling of being alive.",
708: "Two people in a car on a hilltop, silhouetted against a bright orange and pink sunset, with a rainbow-colored sky behind them.",
731: "Car headlights illuminating a winding road, with raindrops resembling snakes slithering across the windshield.",
756: "Two people huddled in the darkness of dusk, illuminated by the car headlights, their faces filled with determination and joy.",
779: "Two people standing in a wintery beach, arms outstretched towards the sky, embracing the moment with a sense of freedom and joy.",
816: "A solitary figure standing on a beach, arms outstretched and face tilted upwards to the sky as the waves crash against their feet and rain showers them with droplets.",
828: "Vibrant sunset silhouetting two figures standing in the surf, arms raised in celebration.",
845: "Person standing on a beach, arms outstretched, face tilted towards the sky, raindrops on their skin.",
864: "A group of people running and jumping into the ocean together at sunset, silhouetted against a backdrop of a vibrant orange and pink sky.",
876: "Vibrant beachgoers standing against a backdrop of crashing waves, beaming with joy.",
893: "Two people emerging from the sea, dripping with water and wearing huge smiles.",
912: "3 carefree people jumping into the waves, silhouetted against a bright orange and pink sunset.",
924: "Vibrant group of people standing in the ocean, laughing and splashing around in the waves despite the cold winter temperatures.",
941: "3 people standing atop a hill overlooking the ocean, arms outstretched, heads tilted back, taking in the cold air and feeling alive.",
960: "3 people standing in the sea, arms outstretched, wind and waves buffeting them as they look to the sky with determination.",
972: "Vibrant sunset reflecting off ocean waves as a group of people stand silhouetted, hands in the air, embracing the moment.",
989: "A person standing at the shoreline with arms raised, waves crashing around them and a sunset in the background.",
}

style_extras = "Vivid watercolor, trending on artstation" #@param {type:"string"}

animation_prompts = {k: v + ", " + style_extras for k, v in raw_animation_prompts.items()}



# %%

from helpers import render


def DeforumArgs():
    #@markdown **Custom Settings**
    override_settings_with_file = False #@param {type:"boolean"}
    settings_file = "custom" #@param ["custom", "512x512_aesthetic_0.json","512x512_aesthetic_1.json","512x512_colormatch_0.json","512x512_colormatch_1.json","512x512_colormatch_2.json","512x512_colormatch_3.json"]
    custom_settings_file = "/content/drive/MyDrive/Settings.txt"#@param {type:"string"}

    #@markdown **Image Settings**
    W = 768 #@param
    H = 768 #@param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64
    bit_depth_output = 8 #@param [8, 16, 32] {type:"raw"}

    #@markdown **Sampling Settings**
    seed = -1 #@param
    sampler = 'euler_ancestral' #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]
    steps = 50 #@param
    scale = 7 #@param
    ddim_eta = 0.0 #@param
    dynamic_threshold = None
    static_threshold = None   

    #@markdown **Save & Display Settings**
    save_samples = True #@param {type:"boolean"}
    save_settings = True #@param {type:"boolean"}
    display_samples = True #@param {type:"boolean"}
    save_sample_per_step = False #@param {type:"boolean"}
    show_sample_per_step = False #@param {type:"boolean"}

    #@markdown **Prompt Settings**
    prompt_weighting = True #@param {type:"boolean"}
    normalize_prompt_weights = True #@param {type:"boolean"}
    log_weighted_subprompts = False #@param {type:"boolean"}

    #@markdown **Batch Settings**
    n_batch = 1 #@param
    batch_name = "" #@param {type:"string"}
    filename_format = "{timestring}_{index}_{prompt}.png" #@param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
    seed_behavior = "iter" #@param ["iter","fixed","random","ladder","alternate"]
    seed_iter_N = 1 #@param {type:'integer'}
    make_grid = False #@param {type:"boolean"}
    grid_rows = 2 #@param 
    outdir = get_output_folder(root.output_path, batch_name)
    print(f"outdir is '{outdir}'")

    #@markdown **Init Settings**
    use_init = False #@param {type:"boolean"}
    strength = 0.1 #@param {type:"number"}
    strength_0_no_init = True # Set the strength to 0 automatically when no init image is used
    init_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg" #@param {type:"string"}
    # Whiter areas of the mask are areas that change more
    use_mask = False #@param {type:"boolean"}
    use_alpha_as_mask = False # use the alpha channel of the init image as the mask
    mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg" #@param {type:"string"}
    invert_mask = False #@param {type:"boolean"}
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = 1.0  #@param {type:"number"}
    mask_contrast_adjust = 1.0  #@param {type:"number"}
    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = True  #@param {type:"boolean"}
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = 5 #@param {type:"number"}

    #@markdown **Exposure/Contrast Conditional Settings**
    mean_scale = 0 #@param {type:"number"}
    var_scale = 0 #@param {type:"number"}
    exposure_scale = 0 #@param {type:"number"}
    exposure_target = 0.5 #@param {type:"number"}

    #@markdown **Color Match Conditional Settings**
    colormatch_scale = 0 #@param {type:"number"}
    colormatch_image = "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png" #@param {type:"string"}
    colormatch_n_colors = 4 #@param {type:"number"}
    ignore_sat_weight = 0 #@param {type:"number"}

    #@markdown **CLIP\Aesthetics Conditional Settings**
    clip_name = 'ViT-L/14' #@param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
    clip_scale = 0 #@param {type:"number"}
    aesthetics_scale = 0 #@param {type:"number"}
    cutn = 1 #@param {type:"number"}
    cut_pow = 0.0001 #@param {type:"number"}

    #@markdown **Other Conditional Settings**
    init_mse_scale = 0 #@param {type:"number"}
    init_mse_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg" #@param {type:"string"}

    blue_scale = 0 #@param {type:"number"}
    
    #@markdown **Conditional Gradient Settings**
    gradient_wrt = 'x0_pred' #@param ["x", "x0_pred"]
    gradient_add_to = 'both' #@param ["cond", "uncond", "both"]
    decode_method = 'linear' #@param ["autoencoder","linear"]
    grad_threshold_type = 'dynamic' #@param ["dynamic", "static", "mean", "schedule"]
    clamp_grad_threshold = 0.2 #@param {type:"number"}
    clamp_start = 0.2 #@param
    clamp_stop = 0.01 #@param
    grad_inject_timing = list(range(1,10)) #@param

    #@markdown **Speed vs VRAM Settings**
    cond_uncond_sync = True #@param {type:"boolean"}

    n_samples = 1 # doesnt do anything
    precision = 'autocast' 
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_sample_raw = None
    mask_sample = None
    init_c = None
    seed_internal = 0

    return locals()



# %%
args_dict = DeforumArgs()
anim_args_dict = DeforumAnimArgs()

args = SimpleNamespace(**args_dict)
anim_args = SimpleNamespace(**anim_args_dict)

if args.override_settings_with_file:
    load_args(args_dict, anim_args_dict, args.settings_file, args.custom_settings_file, verbose=False)



# %%
do_render(args, anim_args, animation_prompts, root)

# give python the hint that we don't need the model any more
root.model = None
gc.collect()
torch.cuda.empty_cache()


# %%
print(f"saved to output folder '{args.outdir}'")

# %% [markdown]
# # Create Video from frames (Google FILM-based frame interpolation)
# 
# [original source](https://www.tensorflow.org/hub/tutorials/tf_hub_film_example)

# %%
# !pip install mediapy
# !sudo apt-get install -y ffmpeg


import tensorflow as tf
import requests
import numpy as np

from typing import Generator, Iterable, List, Optional
import mediapy as media

# %% [markdown]
# ## Various utility functions as part of the Gooogle-provided demo

# %%
_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

def load_image(img_url: str):
  #print(f"load_image: {img_url}")
  """Returns an image with shape [height, width, num_channels], with pixels in [0..1] range, and type np.float32."""

  if (img_url.startswith("https")):
    user_agent = {'User-agent': 'Colab Sample (https://tensorflow.org)'}
    response = requests.get(img_url, headers=user_agent)
    image_data = response.content
  else:
    image_data = tf.io.read_file(img_url)
    #print(f"image_data for '{img_url}': {image_data}")

  image = tf.io.decode_image(image_data, channels=3)
  image_numpy = tf.cast(image, dtype=tf.float32).numpy()
  return image_numpy / _UINT8_MAX_F


# %% [markdown]
# ## Interpolator

# %%
"""A wrapper class for running a frame interpolation based on the FILM model on TFHub

Usage:
  interpolator = Interpolator()
  result_batch = interpolator(image_batch_0, image_batch_1, batch_dt)
  Where image_batch_1 and image_batch_2 are numpy tensors with TF standard
  (B,H,W,C) layout, batch_dt is the sub-frame time in range [0..1], (B,) layout.
"""


def _pad_to_align(x, align):
  """Pads image batch x so width and height divide by align.

  Args:
    x: Image batch to align.
    align: Number to align to.

  Returns:
    1) An image padded so width % align == 0 and height % align == 0.
    2) A bounding box that can be fed readily to tf.image.crop_to_bounding_box
      to undo the padding.
  """
  # Input checking.
  assert np.ndim(x) == 4
  assert align > 0, 'align must be a positive number.'

  height, width = x.shape[-3:-1]
  height_to_pad = (align - height % align) if height % align != 0 else 0
  width_to_pad = (align - width % align) if width % align != 0 else 0

  bbox_to_pad = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height + height_to_pad,
      'target_width': width + width_to_pad
  }
  padded_x = tf.image.pad_to_bounding_box(x, **bbox_to_pad)
  bbox_to_crop = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height,
      'target_width': width
  }
  return padded_x, bbox_to_crop


class Interpolator:
  """A class for generating interpolated frames between two input frames.

  Uses the Film model from TFHub
  """

  def __init__(self, align: int = 64) -> None:
    import tensorflow_hub as hub
    """Loads a saved model.

    Args:
      align: 'If >1, pad the input size so it divides with this before
        inference.'
    """
    self._model = hub.load("https://tfhub.dev/google/film/1")
    self._align = align

  def __call__(self, x0: np.ndarray, x1: np.ndarray,
               dt: np.ndarray) -> np.ndarray:
    """Generates an interpolated frame between given two batches of frames.

    All inputs should be np.float32 datatype.

    Args:
      x0: First image batch. Dimensions: (batch_size, height, width, channels)
      x1: Second image batch. Dimensions: (batch_size, height, width, channels)
      dt: Sub-frame time. Range [0,1]. Dimensions: (batch_size,)

    Returns:
      The result with dimensions (batch_size, height, width, channels).
    """
    if self._align is not None:
      x0, bbox_to_crop = _pad_to_align(x0, self._align)
      x1, _ = _pad_to_align(x1, self._align)

    inputs = {'x0': x0, 'x1': x1, 'time': dt[..., np.newaxis]}
    result = self._model(inputs, training=False)
    image = result['image']

    if self._align is not None:
      image = tf.image.crop_to_bounding_box(image, **bbox_to_crop)
    return image.numpy()

# %%
try: 
  framedir = args.outdir
except NameError: 
  framedir = "/content/drive/MyDrive/AI/StableDiffusion/230508-1945" #@param 


# %%
#@markdown ##Load up the images from the folder 
#@markdown Take the individual frames from deforum's core flow
#@markdown And load them up. 
#@markdown See https://github.com/google-research/frame-interpolation for docs on how to calculate recursion_times. 
#@markdown See also [deforum+FILM sheet](https://docs.google.com/spreadsheets/d/1njAxk9vsavOQH870x369RJ2v6Ax6spGOpjzLmllRj3A/edit#gid=0) for example calc

times_to_interpolate = 3 #@param {type:"slider", min:1, max:10, step:1}


# %% [markdown]
# 

# %% [markdown]
# ## Frame generation

# %%
from tqdm import tqdm

def _recursive_generator(
    frame1: np.ndarray, frame2: np.ndarray, num_recursions: int,
    interpolator: Interpolator) -> Generator[np.ndarray, None, None]:
  """Splits halfway to repeatedly generate more frames.

  Args:
    frame1: Input image 1.
    frame2: Input image 2.
    num_recursions: How many times to interpolate the consecutive image pairs.
    interpolator: The frame interpolator instance.

  Yields:
    The interpolated frames, including the first frame (frame1), but excluding
    the final frame2.
  """
  if num_recursions == 0:
    yield frame1
  else:
    # Adds the batch dimension to all inputs before calling the interpolator,
    # and remove it afterwards.
    time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    mid_frame = interpolator(
        np.expand_dims(frame1, axis=0), np.expand_dims(frame2, axis=0), time)[0]
    yield from _recursive_generator(frame1, mid_frame, num_recursions - 1,
                                    interpolator)
    yield from _recursive_generator(mid_frame, frame2, num_recursions - 1,
                                    interpolator)


def interpolate_recursively(
    frame_filenames: List[str], num_recursions: int,
    interpolator: Interpolator) -> Iterable[np.ndarray]:
  """Generates interpolated frames by repeatedly interpolating the midpoint.

  Args:
    frame_filenames: List of input frame filenames. The colors should be
      in the range[0, 1] and in gamma space.
    num_recursions: Number of times to do recursive midpoint
      interpolation.
    interpolator: The frame interpolation model to use.

  Yields:
    The interpolated frames (including the inputs).
  """
  n = len(frame_filenames)
  for i in tqdm(range(1, n), desc="Processing frames", unit="frames"):
      frame1 = load_image(frame_filenames[i - 1])
      frame2 = load_image(frame_filenames[i])
      yield from _recursive_generator(frame1, frame2,
                                      times_to_interpolate, interpolator)
  # Separately yield the final frame.
  yield load_image(frame_filenames[-1])

import os
from moviepy.editor import concatenate_videoclips, VideoFileClip


def concatenate_videos(video_filenames, output_filename):
    video_clips = [VideoFileClip(video) for video in video_filenames]
    final_video = concatenate_videoclips(video_clips)
    final_video.write_videofile(output_filename)

def generate_video_batches(frame_filenames, recursion_depth, framedir, batch_size):
    interpolator = Interpolator()
    num_batches = (len(frame_filenames) - 1) // batch_size + 1
    batch_filenames = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size - 1, len(frame_filenames) - 1)
        batch_frame_filenames = frame_filenames[start_idx:end_idx + 1]
        frames = list(interpolate_recursively(batch_frame_filenames, recursion_depth, interpolator))
        fps = 24
        batch_movie_filename = framedir + f"-batch{batch_idx}-{fps}fps.mp4"
        print(f'Creating {batch_movie_filename} with {len(frames)} frames')
        media.write_video(batch_movie_filename, frames, fps=fps)
        batch_filenames.append(batch_movie_filename)
    
    return batch_filenames


import glob
filenames = sorted(glob.glob(f"{framedir}/*.png"))
print(f"filenames: {filenames}")

batch_size = 100
batch_filenames = generate_video_batches(filenames, times_to_interpolate, framedir, batch_size)

output_filename = framedir + "-final.mp4"
concatenate_videos(batch_filenames, output_filename)



# %% [markdown]
# # Create Video From Frames (Deforum only)

# %%
# def BasicArgs():
#     skip_video_for_run_all = True #@param {type: 'boolean'}
#     fps = 3 #@param {type:"number"}
#     #@markdown **Manual Settings**
#     use_manual_settings = False #@param {type:"boolean"}
#     image_path = "/content/drive/MyDrive/AI/StableDiffusion/2023-01/StableFun/20230101212135_%05d.png" #@param {type:"string"}
#     mp4_path = "/content/drive/MyDrive/AI/StableDiffusion/2023-01/StableFun/20230101212135.mp4" #@param {type:"string"}
#     render_steps = False  #@param {type: 'boolean'}
#     path_name_modifier = "x0_pred" #@param ["x0_pred","x"]
#     make_gif = False
#     bitdepth_extension = "exr" if args.bit_depth_output == 32 else "png"
#     skip_disconnect_for_run_all = True #@param {type: 'boolean'}
#     max_frames = "60" #@param {type:"string"}
#     return locals()

# %%

# basics = BasicArgs()
# basics = SimpleNamespace(**basics)

# if basics.skip_video_for_run_all == True:
#     print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')
# else:
#     frames2vid(args,anim_args,basics)
# if basics.skip_disconnect_for_run_all == True:
#     print('Skipping disconnect, uncheck skip_disconnect_for_run_all if you want to run it')
# else:
#     from google.colab import runtime
#     runtime.unassign()

# %%



