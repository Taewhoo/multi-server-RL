import getpass
import os

# Set HuggingFace cache directory to use existing cache for models
os.environ['HF_HOME'] = '/raid/taewhoo/.cache/huggingface'


def get_user_tmp():
    user = getpass.getuser()
    # Use a separate directory for AReaL experiment data, not the HuggingFace cache
    user_tmp = "/raid/taewhoo/deep_research/ASearcher/experiments"
    os.makedirs(user_tmp, exist_ok=True)
    return user_tmp
