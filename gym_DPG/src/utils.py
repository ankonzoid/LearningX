"""

 utils.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np

# ==========================
# Check environment
# ==========================

def check_env(env):
    # Make sure action space is discrete (Discrete)
    if 'n' in env.action_space.__dict__:
        pass
    elif 'low' in env.action_space.__dict__ and 'high' in env.action_space.__dict__:
        raise IOError("env.action_space is Box. Stop.")
    else:
        raise IOError("Invalid action space")

    # Make sure observation is continuous (Box)
    if 'n' in env.observation_space.__dict__:
        raise IOError("env.observation_space is Discrete. Stop.")
    elif 'low' in env.observation_space.__dict__ and 'high' in env.observation_space.__dict__:
        pass
    else:
        raise IOError("Invalid observation space")

# ==========================
# General image
# ==========================

def convert_rgb2grey(img_rgb):
    img_grey = np.dot(img_rgb[...,:3], [0.299, 0.587, 0.114])
    return img_grey

def compare_imgs(img1, img2):
    import matplotlib.pyplot as plt
    f, slot = plt.subplots(1, 2)
    slot[0].imshow(img1)
    slot[1].imshow(img2)
    plt.show()
