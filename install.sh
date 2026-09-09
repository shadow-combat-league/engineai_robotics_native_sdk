#!/bin/bash

# Usage: ./install.sh <product_name> <mode>
# Example: ./install.sh pm_v2 robot
# Exits on error
set -e

# !!! Sets the remote user and host
# remote_user="ubuntu"
# remote_host="192.168.69.247"
remote_dir="~/projects/engineai_robotics"

# Env overrides, so the target need not be hand-edited per machine.
# NOTE the deliberate form: these must NOT begin with `remote_host="` at the
# start of a line, because deploy/preflight_hardware.py parses this file with
# ^remote_(user|host|dir)="([^"]*)" and must keep matching the LITERAL
# defaults above rather than a shell expansion.
# SET-BUT-EMPTY IS AN ERROR, not a fallback. With `[ -n "${REMOTE_HOST:-}" ]`
# an empty value silently kept the literal default -- so
# REMOTE_HOST=$(discovery-that-failed) would quietly deploy to the ORIN when
# you meant Nezha. Overwriting the wrong board's config is worse than
# stopping. ${VAR+x} distinguishes unset from empty.
# NOTE: split with parameter expansion, NOT `set --`. `set --` replaces the
# script's positional parameters, and this block runs BEFORE
# product_name="$1" / active_mode="$2" are read -- so it silently installed
# "product REMOTE_DIR, mode remote_dir". Caught by test, 2026-09-09.
for _pair in "REMOTE_USER:remote_user" "REMOTE_HOST:remote_host" "REMOTE_DIR:remote_dir"; do
    _env="${_pair%%:*}"; _var="${_pair##*:}"
    if [ "$(eval "printf '%s' \"\${${_env}+x}\"")" = "x" ]; then
        _val="$(eval "printf '%s' \"\$${_env}\"")"
        if [ -z "$_val" ]; then
            echo "ERROR: ${_env} is set but EMPTY." >&2
            echo "  Refusing to fall back to the built-in default -- that would" >&2
            echo "  deploy to the wrong board silently." >&2
            echo "  If you used \$(deploy/find_robot_via_jump.sh), it failed and" >&2
            echo "  printed nothing; run it alone to see why (stderr)." >&2
            exit 1
        fi
        eval "${_var}=\"\$${_env}\""
    fi
done

# SSH_JUMP: hop through an intermediate host (ssh -J). Added 2026-09-09, when
# the robot's Ethernet died and the only route to the on-robot compute was
# USB-tethering a phone to it and jumping through Termux's sshd:
#
#   SSH_JUMP=u0_a123@10.19.37.133:8022 REMOTE_HOST=192.168.42.2 \
#     ./install.sh t800 robot
#
# Empty by default, so the normal direct path is byte-for-byte unchanged.
ssh_jump_opt=()
ssh_jump_str=""
if [ -n "${SSH_JUMP:-}" ]; then
    ssh_jump_opt=(-J "$SSH_JUMP")
    ssh_jump_str=" -J $SSH_JUMP"
    echo "Jumping via ${SSH_JUMP}"
fi

# Sets the remote destination
# Fail fast on an empty target. Without this, an unset or failed
# REMOTE_HOST (e.g. REMOTE_HOST=$(discovery-script-that-failed)) builds
# "user@" and ssh spends a password prompt and a confusing
# kex_exchange_identification error before dying. Observed 2026-09-09.
for _v in remote_user remote_host remote_dir; do
    if [ -z "$(eval "printf '%s' \"\$$_v\"")" ]; then
        echo "ERROR: $_v is empty." >&2
        echo "  If you used REMOTE_HOST=\$(deploy/find_robot_via_jump.sh)," >&2
        echo "  that script failed and printed nothing -- run it on its own" >&2
        echo "  to see why (its diagnostics go to stderr)." >&2
        exit 1
    fi
done

remote_dest="${remote_user}@${remote_host}"

# Gets the source directory
source_dir=$(cd $(dirname $0) && pwd)
local_install_dir="$source_dir/build/_install"
local_scripts=("$source_dir/scripts/run_robot.sh" "$source_dir/scripts/env_robot.sh" "$source_dir/scripts/set_imu_tty.sh")
local_assets_dir="$source_dir/assets"

# Requires product_name and mode; prints red error and exits if missing
if [ $# -lt 2 ] || [ -z "$1" ] || [ -z "$2" ]; then
    echo -e "\033[31mERROR: product name and mode are required.\033[0m"
    echo -e "\033[31mExample: ./install.sh pm01_edu robot\033[0m"
    exit 1
fi
product_name="$1"
active_mode="$2"
echo "Installing for product: $product_name with mode: $active_mode"

temp_dir=$(mktemp -d)
# Reuses a single SSH connection so password is only prompted once
ssh_control_path="${temp_dir}/ssh_%r_%h_%p"
trap "ssh -o ControlPath=${ssh_control_path} -O exit ${remote_dest} 2>/dev/null || true; rm -rf $temp_dir" EXIT

echo "temp_dir: ${temp_dir}"
rsync -av "${local_install_dir}" "${temp_dir}"
rsync -av "${local_scripts[@]}" "${temp_dir}"

# Copies assets for the given product
if [ -d "${local_assets_dir}/config/$product_name" ]; then
    echo "Copying assets for $product_name..."
    mkdir -p "${temp_dir}/assets/config"
    rsync -av "${local_assets_dir}/config/$product_name" "${temp_dir}/assets/config/"

    # Copy the whole robot resource directory so URDF, meshes and auxiliary XML stay in sync.
    mkdir -p "${temp_dir}/assets/resource/robot"
    if [ -d "${local_assets_dir}/resource/robot/$product_name" ]; then
        rsync -av "${local_assets_dir}/resource/robot/$product_name" "${temp_dir}/assets/resource/robot/"
    fi

    # Copy optional top-level mujoco xml for this product when available.
    mkdir -p "${temp_dir}/assets/resource"
    if [ -f "${local_assets_dir}/resource/${product_name}.xml" ]; then
        rsync -av "${local_assets_dir}/resource/${product_name}.xml" "${temp_dir}/assets/resource/"
    fi
else
    echo "Error: Product directory ${local_assets_dir}/config/$product_name does not exist"
    exit 1
fi

# Add git information if available (兼容普通仓库和 worktree)
git_dir=$( [ -d "$source_dir/.git" ] && echo "$source_dir/.git" || ( [ -f "$source_dir/.git" ] && sed 's/gitdir: //' "$source_dir/.git" ) )
if [ -n "$git_dir" ]; then
    echo "release_note.txt" > "${temp_dir}/release_note.txt"
    git_commit_id=$(git rev-parse HEAD)
    git_branch=$(git rev-parse --abbrev-ref HEAD)
    git_date=$(date "+%Y-%m-%d %H:%M:%S")
    echo "commit_id: ${git_commit_id}" >> "${temp_dir}/release_note.txt"
    echo "branch: ${git_branch}" >> "${temp_dir}/release_note.txt"
    echo "date: ${git_date}" >> "${temp_dir}/release_note.txt"
fi

# Opens a single SSH connection (password prompted once); mkdir and rsync reuse it
echo "Connecting to ${remote_dest} (enter password once)..."
ssh "${ssh_jump_opt[@]}" -o ControlMaster=yes -o ControlPath="${ssh_control_path}" -o ControlPersist=60 "${remote_dest}" "echo 'Connected.'"

# Ensures the remote directory exists (reuses connection, no password)
ssh "${ssh_jump_opt[@]}" -o ControlPath="${ssh_control_path}" "${remote_dest}" "mkdir -p $remote_dir"

rsync -avz --delete -e "ssh -o ControlPath=${ssh_control_path}${ssh_jump_str}" "${temp_dir}/" "${remote_dest}:$remote_dir"

# Updates active_mode in remote mode.yaml (same logic as in scripts/package.sh)
remote_mode_yaml="${remote_dir}/assets/config/${product_name}/mode.yaml"
ssh "${ssh_jump_opt[@]}" -o ControlPath="${ssh_control_path}" "${remote_dest}" "sed -i 's/^active_mode:.*$/active_mode: $active_mode/' $remote_mode_yaml"
echo "Updated active_mode to '$active_mode' in remote $remote_mode_yaml"

echo "Congratulations! Installed to ${remote_user}@${remote_host}:${remote_dir}"
