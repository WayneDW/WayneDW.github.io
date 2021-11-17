# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# If set, the pattern "**" used in a pathname expansion context will
# match all files and zero or more directories and subdirectories.
#shopt -s globstar

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
	# We have color support; assume it's compliant with Ecma-48
	# (ISO/IEC-6429). (Lack of such support is extremely rare, and such
	# a case would tend to support setf rather than setaf.)
	color_prompt=yes
    else
	color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# colored GCC warnings and errors
#export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi



PYTHONDONTWRITEBYTECODE=True
export PYTHONDONTWRITEBYTECODE

alias py='/usr/bin/python'
alias py3='/usr/bin/python3'
alias kp='ps aux | grep python'
alias kr='ps aux | grep Rscript'

alias gpu='nvidia-smi'
alias ll='ls -l'
alias lt='ls -lt -r'
alias r="/usr/bin/r"
alias rm0="find ./ -size  0 -print0 |xargs -0 rm --    "
#alias rm50="find . -name '*' -size -50k -delete "
alias add="git add "
alias cmt="git commit -am "
alias psh="git push"
alias bch="git branch"
alias cout="git checkout"
alias pll="git pull"

set smartindent
set tabstop=4
set shiftwidth=4
set expandtab

# MKL module
export INTEL_COMPILERS_AND_LIBS=/opt/intel/compilers_and_libraries_2019.4.243/linux
source $INTEL_COMPILERS_AND_LIBS/mkl/bin/mklvars.sh intel64

#cd ~/work/Robust-Bayesian-Deep-Active-Learning
#cd /home/deng106/work/Adaptive_Weighted_SG_MCMC
#cd /home/deng106/work/Bayesian-Sparse-Deep-Learning
#cd /home/deng106/work/pytorch_bayes_gan_tentative #/Adaptive_Weighted_SG_MCMC
#cd /home/deng106/work/Quasi_Replica_Exchange_SGMCMC_for_Nonconvex_Learning_in_DNN
#cd /home/deng106/work/Variance-Reduced-Replica-Exchange-Stochastic-Gradient-MCMC
#cd /home/deng106/work/Variance_Reduced_Replica_Exchange_Stochastic_Gradient_MCMC
#cd /home/deng106/work/Sparse_DeepFwFM
#cd /home/deng106/work/AWSGLD #Adaptive_Weighted_SG_MCMC #/Sparse_DeepFwFM/A_Sparse_DeepFM_for_Efficient_CTR_prediction/avazu-ctr-prediction
#cd /home/deng106/work/Contour_SGLD
#cd /home/deng106/work/AWSGLD_Linearized
#cd /home/deng106/work/Pop-CSGLD
cd /home/deng106/work/User_friendly_Replica_Exchange_SGD
#cd /home/deng106/work/AWSGLD_2021

#cd /home/deng106/work/Sparse_DeepFwFM
#cd /home/deng106/work/Sparse_Residual_Network
# don't create .pyc and .pyo files


#more /etc/crontab
#sudo nano /etc/crontab

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

PS1='\n$USER:$PWD\n\$ ' ; export PS1

#PROMPT_COMMAND='echo -en "\033]0;New terminal title\a"' # change title in linux



#virtualenv --system-site-packages -p python ./venv
#source ./venv/bin/activate
#python -m pip install numpy==1.15.0
