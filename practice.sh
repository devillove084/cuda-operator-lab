#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:/opt/conda310/bin:$PATH"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PYTHONDONTWRITEBYTECODE=1
export CUDA_PRACTICE="$root/practice.sh"
python=${PRACTICE_PYTHON:-python3}
if [[ -z ${PRACTICE_PYTHON:-} && -x /opt/conda310/bin/python ]]; then
    python=/opt/conda310/bin/python
fi
if [[ -z ${CUDACXX:-} && -x $CUDA_HOME/bin/nvcc ]]; then
    export CUDACXX="$CUDA_HOME/bin/nvcc"
fi

backend() {
    "$python" "$root/tools/practice_catalog.py" "$@"
}

edit() {
    local file
    local -a editor_command
    file=$(backend source "$@") || return
    read -r -a editor_command <<< "${PRACTICE_EDITOR:-${VISUAL:-${EDITOR:-vi}}}"
    (cd "$root" && "${editor_command[@]}" "$file")
}

menu() {
    if ! command -v fzf >/dev/null || ! command -v less >/dev/null; then
        printf '终端菜单需要 fzf 和 less。可用 ./practice.sh help 查看命令行入口。\n' >&2
        return 1
    fi
    local status=all platform=all query= result menu_code key chosen_platform id description
    local -a selection
    while :; do
        menu_code=0
        result=$(backend list "$status" "$platform" |
            FZF_DEFAULT_OPTS= FZF_DEFAULT_OPTS_FILE= fzf \
                --delimiter=$'\t' --with-nth=3.. --layout=reverse --info=inline-right \
                --border=rounded --color='fg:#d3c6aa,hl:#a7c080,hl+:#a7c080,info:#7fbbb3,prompt:#dbbc7f,pointer:#e69875,border:#475258' --prompt="$platform/$status > " --query="$query" \
                --header='↑↓ 选题 · Enter 题面 · Ctrl-E 编辑 · Ctrl-T 编译测试
Ctrl-B 性能 · Ctrl-X 标记 · Ctrl-S 进度筛选 · Ctrl-P 换题库 · Esc 退出' \
                --preview='"$CUDA_PRACTICE" preview {1} {2}' \
                --preview-window='right,52%,wrap' --print-query \
                --expect=ctrl-e,ctrl-t,ctrl-x,ctrl-s,ctrl-p,ctrl-b) || menu_code=$?
        case "$menu_code" in
            0|1) ;; # fzf returns 1 for filter shortcuts on an empty list.
            130) return 0 ;;
            *) return "$menu_code" ;;
        esac
        mapfile -t selection <<< "$result"
        query=${selection[0]:-}
        key=${selection[1]:-}
        case "$key" in
            ctrl-s)
                case "$status" in all) status=todo ;; todo) status=done ;; done) status=all ;; esac
                continue ;;
            ctrl-p)
                if chosen_platform=$(backend platforms |
                    FZF_DEFAULT_OPTS= FZF_DEFAULT_OPTS_FILE= fzf --layout=reverse \
                        --delimiter=$'\t' --with-nth=2 --prompt='题库 > '); then
                    platform=${chosen_platform%%$'\t'*}
                fi
                continue ;;
        esac
        [[ -n ${selection[2]:-} ]] || continue
        IFS=$'\t' read -r chosen_platform id description <<< "${selection[2]}"
        case "$key" in
            ctrl-e) edit "$chosen_platform" "$id" || pause ;;
            ctrl-t|ctrl-b)
                local action=test
                [[ $key == ctrl-b ]] && action=bench
                printf '\n%s\n' "$description"
                if backend "$action" "$chosen_platform" "$id"; then
                    printf '\n运行结束。'
                else
                    printf '\n编译或测试退出，状态 %s。' "$?"
                fi
                pause ;;
            ctrl-x) backend toggle "$chosen_platform" "$id" || pause ;;
            '') backend show "$chosen_platform" "$id" | less -R || true ;;
        esac
    done
}

pause() {
    printf '\n按 Enter 返回题库 '
    read -r _ </dev/tty || true
}

usage() {
    cat <<'HELP'
./practice.sh                                         打开终端题库
./practice.sh list [all|todo|done] [题库|all]           列出题目与离线测试支持
./practice.sh show|edit|test|toggle 题库 题号           查看、编辑、测试或切换完成状态
./practice.sh cases [题库 [题号]]                      统计用例或查看题目的输入规格
./practice.sh test 题库 题号 --case 1                  重跑指定用例
./practice.sh bench 题库 题号                        对照基线，保存延迟分布
./practice.sh profile 题库 题号 [--tool ncu]             保存 Nsight Systems 时间线
./practice.sh audit                                    检查题库、练习文件和测试入口
./practice.sh roadmap                                  查看完整学习路线
./practice.sh selftest 1 [--plan] [--smoke]                         运行指定阶段的自测

用 ./practice.sh platforms 查看题库名称。LeetCUDA 用函数名或清单显示的编号。
HELP
}

case "${1:-}" in
    '') menu ;;
    help|-h|--help) usage ;;
    edit) shift; edit "$@" ;;
    roadmap) cat "$root/docs/roadmap.md" ;;
    selftest) shift; "$python" "$root/tools/practice_roadmap.py" "$@" ;;
    list|show|test|bench|profile|toggle|preview|audit|source|cases|platforms) backend "$@" ;;
    *) usage >&2; exit 2 ;;
esac
