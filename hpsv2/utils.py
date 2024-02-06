import glob
import os
import subprocess
import requests
import json
import tarfile
from clint.textui import progress
import huggingface_hub

git = os.environ.get('GIT', "git")
environ_root = os.environ.get('HPS_ROOT')
root_path = os.path.expanduser('~/.cache/hpsv2') if environ_root == None else environ_root

hps_version_map = {
    "v2.0": "HPS_v2_compressed.pt",
    "v2.1": "HPS_v2.1_compressed.pt",
}

hps_prompt_files = ['anime.json', 'concept-art.json', 'paintings.json', 'photo.json']

# Model Abbreviations Dict
model_ab_dict = {
        'CM': 'ChilloutMix',
        'Cog2': 'CogView2',
        'DALLE-mini': 'DALL·E mini',
        'DALLE': 'DALL·E 2',
        'DF-IF': 'DeepFloyd-XL',
        'DL': 'Dreamlike Photoreal 2.0',
        'Deliberate': 'Deliberate',
        'ED': 'Epic Diffusion',
        'FD': 'FuseDream',
        'LDM': 'Latent Diffusion',
        'Laf': 'LAFITE',
        'MM': 'MajicMix Realistic',
        'OJ': 'Openjourney',
        'RV': 'Realistic Vision',
        'SDXL-base-0.9': 'SDXL Base 0.9',
        'SDXL-refiner-0.9': 'SDXL Refiner 0.9',
        'VD': 'Versatile Diffusion',
        'VQD': 'VQ-Diffusion',
        'VQGAN': 'VQGAN + CLIP',
        'glide': 'GLIDE',
        'sdv1': 'Stable Diffusion v1.4',
        'sdv2': 'Stable Diffusion v2.0'
    }

def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    if desc is not None:
        print(desc)

    if live:
        result = subprocess.run(command, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")

def git_clone(url, dir, name, commithash=None):
    # TODO clone into temporary dir and move if successful

    if os.path.exists(dir):
        if commithash is None:
            return

        current_hash = run(f'"{git}" -C "{dir}" rev-parse HEAD', None, f"Couldn't determine {name}'s hash: {commithash}").strip()
        if current_hash == commithash:
            return

        run(f'"{git}" -C "{dir}" fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}")
        run(f'"{git}" -C "{dir}" checkout {commithash}', f"Checking out commit for {name} with hash: {commithash}...", f"Couldn't checkout commit {commithash} for {name}")
        return

    run(f'"{git}" clone "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}")

    if commithash is not None:
        run(f'"{git}" -C "{dir}" checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")

def load_models(model_path: str, model_url: str = None, command_path: str = None, ext_filter=None, download_name=None, ext_blacklist=None) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.

    @param download_name: Specify to download from model_url immediately.
    @param model_url: If no other models are found, this will be downloaded on upscale.
    @param model_path: The location to store/find models in.
    @param command_path: A command-line argument to search for models in first.
    @param ext_filter: An optional list of filename extensions to filter by
    @return: A list of paths containing the desired model(s)
    """
    output = []

    if ext_filter is None:
        ext_filter = []

    try:
        places = []

        if command_path is not None and command_path != model_path:
            pretrained_path = os.path.join(command_path, 'experiments/pretrained_models')
            if os.path.exists(pretrained_path):
                print(f"Appending path: {pretrained_path}")
                places.append(pretrained_path)
            elif os.path.exists(command_path):
                places.append(command_path)

        places.append(model_path)

        for place in places:
            if os.path.exists(place):
                for file in glob.iglob(place + '**/**', recursive=True):
                    full_path = file
                    if os.path.isdir(full_path):
                        continue
                    if os.path.islink(full_path) and not os.path.exists(full_path):
                        print(f"Skipping broken symlink: {full_path}")
                        continue
                    if ext_blacklist is not None and any([full_path.endswith(x) for x in ext_blacklist]):
                        continue
                    if len(ext_filter) != 0:
                        model_name, extension = os.path.splitext(file)
                        if extension not in ext_filter:
                            continue
                    if file not in output:
                        output.append(full_path)

        if model_url is not None and len(output) == 0:
            if download_name is not None:
                from basicsr.utils.download_util import load_file_from_url
                dl = load_file_from_url(model_url, model_path, True, download_name)
                output.append(dl)
            else:
                output.append(model_url)

    except Exception:
        pass

    return output


def download_benchmark_prompts() -> None:
    
    folder_name = os.path.join(root_path, 'datasets/benchmark')
    os.makedirs(folder_name, exist_ok=True)
    for file in hps_prompt_files:
        file_name = huggingface_hub.hf_hub_download("zhwang/HPDv2", file, subfolder="benchmark", repo_type="dataset")
        if not os.path.exists(os.path.join(folder_name, file)):
            os.symlink(file_name, os.path.join(folder_name, file))
    # huggingface_hub.hf_hub_download("zhwang/HPDv2", "concept-art.json", subfolder="benchmark", repo_type="dataset")
    # huggingface_hub.hf_hub_download("zhwang/HPDv2", "paintings.json", subfolder="benchmark", repo_type="dataset")
    # huggingface_hub.hf_hub_download("zhwang/HPDv2", "photo.json", subfolder="benchmark", repo_type="dataset")


def download_benchmark_images(model_id: str) -> None:
    try:
        i = list(model_ab_dict.values()).index(model_id)
        model_id = list(model_ab_dict.keys())[i]
    except ValueError:
        print('Input model not in model dict.')
        if model_id not in model_ab_dict.keys():
            return
    
    file_name = huggingface_hub.hf_hub_download("zhwang/HPDv2", model_id + '.tar.gz', subfolder="benchmark/benchmark_imgs", repo_type="dataset")
    print(file_name)
    folder_name = os.path.join(root_path, 'datasets/benchmark/benchmark_imgs/'+model_id)
    
    if not os.path.exists(folder_name):
        with tarfile.open(file_name, 'r:*') as tar:
            tar.extractall(path=os.path.join(root_path, 'datasets/benchmark/benchmark_imgs/'))
        print('Extract /'+model_id+' successfully!')
        
    return
