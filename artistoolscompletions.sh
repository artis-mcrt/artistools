#!/usr/bin/env zsh

# Run something, muting output or redirecting it to the debug stream
# depending on the value of _ARC_DEBUG.
# If ARGCOMPLETE_USE_TEMPFILES is set, use tempfiles for IPC.
__python_argcomplete_run() {
    if [[ -z "${ARGCOMPLETE_USE_TEMPFILES-}" ]]; then
        __python_argcomplete_run_inner "$@"
        return
    fi
    local tmpfile="$(mktemp)"
    _ARGCOMPLETE_STDOUT_FILENAME="$tmpfile" __python_argcomplete_run_inner "$@"
    local code=$?
    cat "$tmpfile"
    rm "$tmpfile"
    return $code
}

__python_argcomplete_run_inner() {
    if [[ -z "${_ARC_DEBUG-}" ]]; then
        "$@" 8>&1 9>&2 1>/dev/null 2>&1
    else
        "$@" 8>&1 9>&2 1>&9 2>&1
    fi
}

_python_argcomplete() {
    local IFS=$'\013'
    if [[ -n "${ZSH_VERSION-}" ]]; then
        local completions
        completions=($(IFS="$IFS" \
            COMP_LINE="$BUFFER" \
            COMP_POINT="$CURSOR" \
            _ARGCOMPLETE=1 \
            _ARGCOMPLETE_SHELL="zsh" \
            _ARGCOMPLETE_SUPPRESS_SPACE=1 \
            __python_argcomplete_run "${words[1]}") )
        _describe "${words[1]}" completions -o nosort
    else
        local SUPPRESS_SPACE=0
        if compopt +o nospace 2> /dev/null; then
            SUPPRESS_SPACE=1
        fi
        COMPREPLY=($(IFS="$IFS" \
            COMP_LINE="$COMP_LINE" \
            COMP_POINT="$COMP_POINT" \
            COMP_TYPE="$COMP_TYPE" \
            _ARGCOMPLETE_COMP_WORDBREAKS="$COMP_WORDBREAKS" \
            _ARGCOMPLETE=1 \
            _ARGCOMPLETE_SHELL="bash" \
            _ARGCOMPLETE_SUPPRESS_SPACE=$SUPPRESS_SPACE \
            __python_argcomplete_run "$1"))
        if [[ $? != 0 ]]; then
            unset COMPREPLY
        elif [[ $SUPPRESS_SPACE == 1 ]] && [[ "${COMPREPLY-}" =~ [=/:]$ ]]; then
            compopt -o nospace
        fi
    fi
}


if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete at
else
    compdef _python_argcomplete at
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools
else
    compdef _python_argcomplete artistools
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-comparetogsinetwork
else
    compdef _python_argcomplete artistools-comparetogsinetwork
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-modeldeposition
else
    compdef _python_argcomplete artistools-modeldeposition
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete getartisspencerfano
else
    compdef _python_argcomplete getartisspencerfano
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-spencerfano
else
    compdef _python_argcomplete artistools-spencerfano
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete listartistimesteps
else
    compdef _python_argcomplete listartistimesteps
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-timesteptimes
else
    compdef _python_argcomplete artistools-timesteptimes
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-make1dslicefrom3dmodel
else
    compdef _python_argcomplete artistools-make1dslicefrom3dmodel
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodel1dslicefromcone
else
    compdef _python_argcomplete makeartismodel1dslicefromcone
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelbotyanski2017
else
    compdef _python_argcomplete makeartismodelbotyanski2017
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelfromshen2018
else
    compdef _python_argcomplete makeartismodelfromshen2018
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelfromlapuente
else
    compdef _python_argcomplete makeartismodelfromlapuente
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelscalevelocity
else
    compdef _python_argcomplete makeartismodelscalevelocity
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelfullymixed
else
    compdef _python_argcomplete makeartismodelfullymixed
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelsolar_rprocess
else
    compdef _python_argcomplete makeartismodelsolar_rprocess
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelfromsingletrajectory
else
    compdef _python_argcomplete makeartismodelfromsingletrajectory
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelfromparticlegridmap
else
    compdef _python_argcomplete makeartismodelfromparticlegridmap
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodel
else
    compdef _python_argcomplete makeartismodel
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-maketardismodelfromartis
else
    compdef _python_argcomplete artistools-maketardismodelfromartis
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-maptogrid
else
    compdef _python_argcomplete artistools-maptogrid
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartismodeldensity
else
    compdef _python_argcomplete plotartismodeldensity
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-plotdensity
else
    compdef _python_argcomplete artistools-plotdensity
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartismodeldeposition
else
    compdef _python_argcomplete plotartismodeldeposition
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-deposition
else
    compdef _python_argcomplete artistools-deposition
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-describeinputmodel
else
    compdef _python_argcomplete artistools-describeinputmodel
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartisestimators
else
    compdef _python_argcomplete plotartisestimators
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-estimators
else
    compdef _python_argcomplete artistools-estimators
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-exportmassfractions
else
    compdef _python_argcomplete artistools-exportmassfractions
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartislightcurve
else
    compdef _python_argcomplete plotartislightcurve
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-lightcurve
else
    compdef _python_argcomplete artistools-lightcurve
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartislinefluxes
else
    compdef _python_argcomplete plotartislinefluxes
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-linefluxes
else
    compdef _python_argcomplete artistools-linefluxes
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartismacroatom
else
    compdef _python_argcomplete plotartismacroatom
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-macroatom
else
    compdef _python_argcomplete artistools-macroatom
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartisnltepops
else
    compdef _python_argcomplete plotartisnltepops
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-nltepops
else
    compdef _python_argcomplete artistools-nltepops
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartisnonthermal
else
    compdef _python_argcomplete plotartisnonthermal
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-nonthermal
else
    compdef _python_argcomplete artistools-nonthermal
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartisradfield
else
    compdef _python_argcomplete plotartisradfield
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-radfield
else
    compdef _python_argcomplete artistools-radfield
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartisspectrum
else
    compdef _python_argcomplete plotartisspectrum
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-spectrum
else
    compdef _python_argcomplete artistools-spectrum
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartistransitions
else
    compdef _python_argcomplete plotartistransitions
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-transitions
else
    compdef _python_argcomplete artistools-transitions
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartisinitialcomposition
else
    compdef _python_argcomplete plotartisinitialcomposition
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-initialcomposition
else
    compdef _python_argcomplete artistools-initialcomposition
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-writecodecomparisondata
else
    compdef _python_argcomplete artistools-writecodecomparisondata
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-setup_completions
else
    compdef _python_argcomplete artistools-setup_completions
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-viewingangles
else
    compdef _python_argcomplete artistools-viewingangles
fi

if [[ -z "${ZSH_VERSION-}" ]]; then
    complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartisviewingangles
else
    compdef _python_argcomplete plotartisviewingangles
fi

