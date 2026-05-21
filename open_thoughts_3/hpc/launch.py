import os
import re
import yaml
import wandb
import subprocess
import dataclasses

from collections import defaultdict

from huggingface_hub import snapshot_download
from datasets import load_dataset

from hpc.arguments import LlamaFactoryArgs, parse_args
from hpc.hpc import detect_hpc, set_environment

def check_exists(local_path):
    if os.path.exists(local_path):
        return True
    else:
        return False

def launch_sbatch(sbatch_script_path, dependency=None) -> str:
    sbatch_cmd = ["sbatch"]
    if dependency is not None:
        sbatch_cmd.append(f"--dependency={dependency}")
    sbatch_cmd.append(sbatch_script_path)

    job_id = subprocess.check_output(sbatch_cmd).decode("utf-8").strip()
    print(f"Job {job_id} submitted with dependency {dependency}.")
    return job_id


def wandb_init(kwargs):
    wandb_run_name = "_".join([str(value) for key, value in kwargs.items()])
    wandb_run_name = wandb_run_name.replace("/", "_")
    wandb_project = os.path.expandvars(os.environ.get("WANDB_PROJECT", "dcft"))
    wandb.init(project=wandb_project, name=wandb_run_name, config=kwargs)


def get_job_name(cli_args):
    job_name_components = []
    job_name_suffix = None
    for key, value in cli_args.items():
        if (
            (
                isinstance(value, str)
                or isinstance(value, int)
                or isinstance(value, float)
            )
            and value not in ["None"]
            and key
            not in [
                "name",
                "dotenv_filename",
                "dotenv_path",
                "train_sbatch_filename",
                "hostname",
                "deepspeed",
                "train_config_path_out",
                "train_config_path",
                "hostname_pattern",
                "num_nodes",
                "gpus_per_node",
                "cpus_per_task",
                "time_limit",
                "output_dir",
                "experiments_dir",
                "train_sbatch_path",
                "train_sbatch_path_out",
                "partition",
                "account",
                "gpus_type",
                "total_partition_nodes",
                "internet_node",
                "mem_per_node",
                "cpus_per_node",
                "role_tag",
                "user_tag",
                "messages",
                "system",
                "assistant_tag",
                "content_tag",
                "eval_tasks",
                "eval_num_nodes",
                "eval_time_limit",
                "qos",
                "max_restarts",
                "pretokenize",
                "pretok_large",
                "dry_run"
            ]
        ):
            if key not in ["dataset", "model_name_or_path"]:
                job_name_components.append(str(key.replace("_", "-")))

            if value == "Qwen/Qwen2.5-32B-Instruct":
                job_name_suffix = "_32B"
            elif value == "Qwen/Qwen2.5-14B-Instruct":
                job_name_suffix = "_14B"
            elif value == "Qwen/Qwen2.5-3B-Instruct":
                job_name_suffix = "_3B"
            elif value == "Qwen/Qwen2.5-1.5B-Instruct":
                job_name_suffix = "_1.5B"
            else:
                job_name_components.append(str(value).split("/")[-1])
    job_name = "_".join(job_name_components)
    job_name = (
        job_name.replace("/", "_")
        .replace("?", "")
        .replace("*", "")
        .replace(" ", "_")
    )
    if job_name_suffix is not None:
        job_name += job_name_suffix
    # truncate job name to 96 characters
    if len(job_name) > 96:
        print("Truncating job name to less than HF limit of 96 characters...")
        job_name = "_".join(
            "-".join(y[:4] for y in x.split("-")) for x in job_name.split("_")
        )
        if len(job_name) > 96:
            raise ValueError(
                f"Job name {job_name} is still too long (96 characters) after truncation. Try renaming the dataset or making a yaml with the config."
            )

    return job_name


# Curly braces but not those within ${...}
curly_brace_pattern = r"(?<!\$)\{([^{}]*)\}"


def extract_template_keys(file_path):
    with open(file_path, "r") as f:
        file = f.read()
    return re.findall(curly_brace_pattern, file)


def fill_template(file_path, exp_args, new_file_path):
    with open(file_path, "r") as f:
        file = f.read()

    file = re.sub(curly_brace_pattern, lambda m: exp_args[m.group(1)], file)

    with open(new_file_path, "w") as f:
        f.write(file)


def construct_sbatch_script(exp_args):
    base_script_path = exp_args["train_sbatch_path"]
    with open(base_script_path, "r") as f:
        base_script = f.read()

    kwargs = defaultdict(str, **exp_args)

    # find JSON file creation with cat
    json_files_cat = re.findall(r"cat.*?<<EOT >.*?EOT", base_script, re.DOTALL)
    json_filenames = []
    for json_file in json_files_cat:
        json_file_name = re.match(
            r"cat.*?<<EOT >.*?(\S+).*?EOT", json_file, re.DOTALL
        ).group(1)
        json_filenames.append(json_file_name)

        base_script = re.sub(
            r"cat.*?<<EOT >.*?" + json_file_name.replace("$", "\\$") + r".*?EOT",
            f"cat {json_file_name}",
            base_script,
            count=1,
            flags=re.DOTALL,
        )

    # safeguard against injection of bash ${} variables
    bash_variables = re.findall(r"\${.*?}", base_script)
    for var in bash_variables:
        base_script = base_script.replace(
            var, var.replace("{", "{{").replace("}", "}}")
        )

    time_limit = kwargs.get("time_limit")
    if time_limit is None:
        time_limit = "01:00:00"
        kwargs["time_limit"] = time_limit

    sbatch_script = base_script.format(**kwargs)

    for json_file, json_file_name in zip(json_files_cat, json_filenames):
        sbatch_script = sbatch_script.replace(f"cat {json_file_name}", json_file)

    sbatch_dir = os.path.join(kwargs["experiments_dir"], "sbatch_scripts")
    os.makedirs(sbatch_dir, exist_ok=True)
    sbatch_script_path = os.path.join(sbatch_dir, f"{kwargs['job_name']}.sbatch")
    with open(sbatch_script_path, "w") as f:
        f.write(sbatch_script)
        print(f"Wrote sbatch script to {sbatch_script_path}")

    return sbatch_script_path


def construct_config_yaml(exp_args):
    configs_dir = os.path.join(exp_args["experiments_dir"], "configs")
    os.makedirs(configs_dir, exist_ok=True)

    train_config_path = exp_args.get("train_config_path")
    checkpoints_dir = exp_args.get("checkpoints_dir")
    models_dir = exp_args.get("models_dir")
    datasets_dir = exp_args.get("datasets_dir")

    datasets_dir = os.path.expandvars(os.environ.get("DATASETS_DIR", datasets_dir))
    models_dir = os.path.expandvars(os.environ.get("MODELS_DIR", models_dir))
    checkpoints_dir = os.path.expandvars(
        os.environ.get("CHECKPOINTS_DIR", checkpoints_dir)
    )

    os.makedirs(checkpoints_dir, exist_ok=True)
    with open(train_config_path, "r") as f:
        base_config = f.read()

    # don't do templating for the yaml - simplification
    # base_config = base_config.format(**exp_args)

    base_config = yaml.safe_load(base_config)

    # Update base config with experiment arguments
    for key, value in exp_args.items():
        if key in base_config or key in [
            field.name for field in dataclasses.fields(LlamaFactoryArgs)
        ]:
            print(f"Setting {key} to {value}")
            base_config[key] = value

    if base_config.get("dataset_dir") is None:
        base_config["dataset_dir"] = "ONLINE"

    if "_pretokenize" in exp_args["job_name"]:
        # Already have downloaded dataset and model, since the train yaml is already constructed (a bit hacky)
        model_path = exp_args["model_name_or_path"]
        dataset_path = exp_args["dataset"]
    else:
        # Download Dataset and Model - MAKE SURE HF_HUB_CACHE is set!
        dataset_path = snapshot_download(repo_id=base_config["dataset"], repo_type="dataset")
        print(f"Downloaded dataset to {dataset_path}")
        if not exp_args["internet_node"]:
            # This needs to be done on the login node for passing the repo_id name and "cache_dir" to llamafactory
            # this is a result / another quirk of using datasets 3.1 instead of upgrading (which can't be done with our current version of llamafactory)
            # even with pretokenization, this is necessary since the pretokenizer on the compute nodes doesn't have access to the internet and won't see this dataset unless it is already loaded
            ds = load_dataset(path=base_config["dataset"], cache_dir=os.environ["HF_HUB_CACHE"]) # note this might be slow on the login node for big datasets 
        model_path = snapshot_download(repo_id=base_config["model_name_or_path"], repo_type="model")
        print(f"Downloaded model to {model_path}")

    if base_config.get("output_dir") and checkpoints_dir not in base_config.get(
        "output_dir"
    ):
        base_config["output_dir"] = os.path.join(
            checkpoints_dir, base_config["output_dir"]
        )
    else:
        base_config["output_dir"] = os.path.join(checkpoints_dir, exp_args["job_name"])
    os.makedirs(base_config["output_dir"], exist_ok=True)

    wandb_dir = os.path.join(exp_args["experiments_dir"], "wandb", exp_args["job_name"])
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir

    hub_model_id = base_config.get("hub_model_id", None)
    if hub_model_id is not None:
        hub_model_id = hub_model_id.replace(".", "_")
    base_config["hub_model_id"] = hub_model_id

    if exp_args["internet_node"]:
        base_config["report_to"] = "wandb"
        base_config["push_to_db"] = True
        base_config["push_to_hub"] = True
    else:
        # no wandb reporting
        if "report_to" in base_config:
            del base_config["report_to"]
        # you need to explicitly set these to false
        base_config["push_to_db"] = False
        base_config["push_to_hub"] = False
        # https://github.com/mlfoundations/dcft_private/blob/04cd3e26937803aecb051db4b304dd2f4ac1808f/dcft/train/leonardo/train.py#L463-L470
        base_config["datasets_cache_dir"] = os.environ["HF_HUB_CACHE"] # llama factory sets this as cache_dir in `load_dataset`
        # config['dataset'] = dataset_path # this only works with newer datasets package (3.3+) which is not compatible with this version of llamafactory
        # therefore, we use the above method where the dataset HF repo id is provided and we rely on setting the cache_dir when loading
        base_config["dataset_dir"] = "ONLINE"  # this just means the dataset is in HF format
        # we need to pass directly the downloaded model path in the cache (there is no setting for model_cache_dirs)
        base_config["model_name_or_path"] = model_path

    num_nodes = int(exp_args.get("num_nodes"))
    num_gpus = exp_args.get("gpus_per_node")
    global_batch_size = int(base_config.get("global_batch_size"))
    print(
        f"\nCalculated based on {num_nodes} nodes, {num_gpus} GPUs per node, and global batch size {global_batch_size}:"
    )
    if global_batch_size is not None:
        per_device_train_batch_size = 1
        gradient_accumulation_steps = global_batch_size // (
            per_device_train_batch_size * num_nodes * num_gpus
        )
        base_config["gradient_accumulation_steps"] = gradient_accumulation_steps
        base_config["per_device_train_batch_size"] = per_device_train_batch_size
        print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
        print(f"per_device_train_batch_size: {per_device_train_batch_size}")
    hub_model_id = base_config.get("hub_model_id", None)

    if hub_model_id is None:
        hub_model_id = "mlfoundations-dev/" + exp_args["job_name"]
    base_config["hub_model_id"] = hub_model_id
    tokenized_path = base_config.get("tokenized_path")

    if tokenized_path is None and exp_args.get("pretokenize"):
        tokenized_dir = exp_args.get("tokenized_dir")
        tokenized_dir = os.path.expandvars(
        os.environ.get("TOKENIZED_DATASETS_DIR", tokenized_dir)
        )
        model_name = "_".join(
            base_config["model_name_or_path"].split("/")[-2:]
        ).replace(".", "-")
        dataset_name = "_".join(base_config.get("dataset").split("/")[-2:])
        base_config["tokenized_path"] = os.path.join(
            tokenized_dir, "_".join([dataset_name, model_name, "tokenized"])
        )
        exp_args["tokenized_path"] = base_config["tokenized_path"]

    data_tags = [
        "role_tag",
        "content_tag",
        "assistant_tag",
        "user_tag",
        "messages",
        "system",
    ]

    for tag in data_tags:
        if tag in exp_args:
            tag_value = exp_args[tag]
            if tag_value is not None:
                base_config[tag] = tag_value

    train_config_path_out = os.path.join(
        configs_dir, exp_args["job_name"] + "_train_config.yaml"
    )
    with open(train_config_path_out, "w") as f:
        yaml.dump(base_config, f)
        print(f"Wrote config to {train_config_path_out}")

    exp_args["output_dir"] = base_config["output_dir"]
    exp_args["dataset"] = base_config["dataset"]
    exp_args["model_name_or_path"] = base_config["model_name_or_path"]
    exp_args["hub_model_id"] = base_config.get("hub_model_id", None)
    return base_config, train_config_path_out


def submit_job(
    exp_args=None,
    dependency=None,
):
    exp_args["logs_dir"] = os.path.join(exp_args["experiments_dir"], "logs")
    os.makedirs(exp_args["logs_dir"], exist_ok=True)

    job_id = None
    if exp_args.get("max_restarts") is not None:
        max_restarts = int(exp_args["max_restarts"])
        if max_restarts > 0:
            for _ in range(max_restarts):
                job_id = launch_sbatch(
                    exp_args["train_sbatch_path_out"], dependency=dependency
                )
                job_id = job_id.split()[-1]
                dependency = f"afternotok:{job_id}"

    job_id = launch_sbatch(
        exp_args["train_sbatch_path_out"], dependency
    )
    job_id = job_id.split()[-1]
    print(f"Writing logs to {exp_args['logs_dir']}/{exp_args['job_name']}_{job_id}.out")
    return job_id


def update_exp_args(exp_args, args):
    for key, value in args.items():
        if key in exp_args and value is None:
            del exp_args[key]
            print(f"Removed {key} from experiment arguments")
        elif key in exp_args and value != exp_args[key]:
            print(f"Overwrote {key} from {exp_args[key]} to {value}")
        exp_args[key] = value
    return exp_args


def display_args(exp_args, name):
    print()
    print("=" * 20 + f" {name} Args " + "=" * 20)
    for key, value in exp_args.items():
        print(f"{key}: {value}")
    print()


def pre_validation(exp_args, cli_args):

    # Add arguments to experiment from train config file
    if "train_config_path" in cli_args and os.path.exists(
        cli_args["train_config_path"]
    ):
        # with open(cli_args["train_config_path"], "r") as f:
        #     config = yaml.safe_load(f)
        #     exp_args = update_exp_args(exp_args, config)
        pass
    elif "train_config_path" in cli_args:
        raise FileNotFoundError(
            f"Train config file {cli_args['train_config_path']} does not exist."
        )

    # Fill in sbatch template
    if "train_sbatch_path" in exp_args and os.path.exists(
        exp_args["train_sbatch_path"]
    ):
        template_keys = extract_template_keys(exp_args["train_sbatch_path"])
        for key in template_keys:
            if (
                key not in exp_args
                and key not in cli_args
                and key not in ["train_config_path_out"]
            ):
                raise ValueError(
                    f"Template key {key} not found in experiment arguments or cli arguments."
                )
    elif "train_sbatch_path" in exp_args:
        raise FileNotFoundError(
            f"Train sbatch file {exp_args['train_sbatch_path']} does not exist."
        )

def schedule_eval(exp_args, train_job_id):
    eval_tasks = exp_args["eval_tasks"]
    model_name = f"mlfoundations-dev/{exp_args['job_name']}"
    
    num_nodes = exp_args.get("eval_num_nodes")
    if num_nodes is None:
        num_nodes = os.environ["NUM_NODES_DEFAULT"]
    num_shards = str(int(num_nodes) * int(os.environ["NUM_GPUS_PER_NODE"]))

    eval_time_limit = exp_args.get("eval_time_limit", "4:00:00")
    max_job_duration = str(int(eval_time_limit.split(":")[0]))

    evalchemy_path = os.environ["EVALCHEMY"]
    evalchemy_activate_env = os.environ["EVALCHEMY_ACTIVATE_ENV"]
    
    print(f"Scheduling automatic evalution following training job:")
    if eval_tasks:
        eval_cmd = f"{evalchemy_activate_env}"
        eval_cmd += f" && cd {evalchemy_path}"
        eval_cmd += f" && python eval/distributed/launch_simple.py"
        eval_cmd += f" --tasks {eval_tasks}"
        eval_cmd += f" --num_shards {num_shards}"
        eval_cmd += f" --max-job-duration {max_job_duration}"
        eval_cmd += f" --model_name {model_name}"
        eval_cmd += f" --dependency afterok:{train_job_id}"
        
        print(f"Launching evaluation with command: {eval_cmd}")
        result = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            # Filter out Vista system checks using regex
            filtered_output = re.sub(r'-+\n\s+Welcome to.*\n-+\n.*?(Submitted batch job \d+)', r'\1', result.stdout, flags=re.DOTALL)
            filtered_output = re.sub(r'No reservation.*\n(-->.*\n)*', '', filtered_output)
            print(filtered_output.strip())
        else:
            print("Error launching evaluation job:")
            print(result.stderr)

def schedule_pretokenize(exp_args):
    pretok_args = exp_args.copy()
    pretok_args = update_exp_args(pretok_args, {"job_name": f"{exp_args['job_name']}_pretokenize"})
    # this is needed because otherwise alueError: training_args.world_size 4 * training_args.per_device_train_batch_size 1 * training_args.gradient_accumulation_steps 1 needs to equal model_args.global_batch_size 512
    pretok_args = update_exp_args(pretok_args, {"num_nodes": 1})
    # pretok_args = update_exp_args(pretok_args, {"deepspeed": None, "enable_liger_kernel": False, })
    # Just use the same LF yaml for the pretokenization job
    pretok_train_config, pretok_train_config_path_out = construct_config_yaml(pretok_args)
    if exp_args.get("pretok_large"):
        # You shouldn't pretokenize on 128 nodes
        # pretok_args = update_exp_args(pretok_args, {"num_nodes": 128, "qos": "boost_qos_bprod", "time_limit": "1-00:00:00", "max_restarts": 0, "job_name": f"{exp_args['job_name']}_pretokenize"})
        if exp_args["name"] != "leonardo":
            raise ValueError("Large pretokenization is only supported on leonardo")
        pretok_args = update_exp_args(pretok_args, {
        "time_limit": "03:00:00",
        "qos": "normal",
        "max_restarts": 0,
        "node_exclusion_list": "",
        "job_name": f"{exp_args['job_name']}_pretokenize"})
    else:
        pretok_args = update_exp_args(pretok_args, {
        "partition": exp_args['pretok_partition'], 
        "qos": exp_args['pretok_qos'], 
        "time_limit": exp_args['pretok_time_limit'],
        # this I was using to test with cpu only nodes - needed to make the srun work
        # "cpus_per_node": exp_args['pretok_cpus_per_node'],
        # "gpus_per_node": exp_args['pretok_gpus_per_node'],
        "max_restarts": 0,
        })
    pretok_args = update_exp_args(pretok_args, pretok_train_config)
    pretok_args = update_exp_args(pretok_args, {"train_config_path_out": pretok_train_config_path_out})
    pretok_sbatch_path_out = construct_sbatch_script(pretok_args)
    pretok_args = update_exp_args(pretok_args, {"train_sbatch_path_out": pretok_sbatch_path_out})
    pretok_job_id = submit_job(
        exp_args=pretok_args,
        dependency=None,
    )
    return pretok_job_id

def main():
    print()
    # this is where defaults are stored for experiments_dir and deepspeed
    cli_args = parse_args()
    for key, value in cli_args.items():
        if type(value) == str:
            value = value.lower()
            if value == "false":
                cli_args[key] = False
            elif value == "true":
                cli_args[key] = True
            elif value == "none":
                cli_args[key] = None

    # Storing all the arguments in a dictionary that we add to in order of precedence
    exp_args = dict()

    # Add arguments to experiment from automatically detecting HPC
    hpc = detect_hpc()
    set_environment(hpc)

    # Add arguments and validate
    print()
    exp_args = update_exp_args(exp_args, hpc.model_dump())
    exp_args = update_exp_args(exp_args, cli_args)

    # Job name
    if "job_name" not in exp_args:
        exp_args["job_name"] = get_job_name(cli_args)
    print(f"Job name: {exp_args['job_name']}")

    # Pre-validation
    pre_validation(exp_args, cli_args)

    # Construct the config yaml
    print()
    train_config, train_config_path_out = construct_config_yaml(exp_args)
    exp_args = update_exp_args(exp_args, train_config)
    exp_args = update_exp_args(
        exp_args, {"train_config_path_out": train_config_path_out}
    )

    # Construct the sbatch script
    print()
    train_sbatch_path_out = construct_sbatch_script(exp_args)
    exp_args = update_exp_args(
        exp_args, {"train_sbatch_path_out": train_sbatch_path_out}
    )

    display_args(exp_args, "Train")
    if exp_args.get("dry_run", False):
        print(
            "DRY RUN: Job would be submitted with the above parameters, but --dry_run flag was set."
        )
    else:
        dependency = None
        if exp_args.get("pretokenize"):
            if os.path.exists(exp_args["tokenized_path"]):
                print(f"Tokenized directory {exp_args['tokenized_path']} already exists, skipping pretokenization job submission")
            else:
                pretok_job_id = schedule_pretokenize(exp_args)
                dependency = f"afterok:{pretok_job_id}"

        train_job_id = submit_job(
            exp_args=exp_args,
            dependency=dependency,
        )

        if exp_args.get("eval_tasks"):
            if exp_args.get("internet_node", False):
                print()
                schedule_eval(exp_args, train_job_id)
            else:
                print("Skipping evaluation because internet_node is False")

if __name__ == "__main__":
    main()
