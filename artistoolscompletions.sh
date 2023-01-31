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
    local SUPPRESS_SPACE=0
    if compopt +o nospace 2> /dev/null; then
        SUPPRESS_SPACE=1
    fi
    COMPREPLY=( $(IFS="$IFS" \
                  COMP_LINE="$COMP_LINE" \
                  COMP_POINT="$COMP_POINT" \
                  COMP_TYPE="$COMP_TYPE" \
                  _ARGCOMPLETE_COMP_WORDBREAKS="$COMP_WORDBREAKS" \
                  _ARGCOMPLETE=1 \
                  _ARGCOMPLETE_SUPPRESS_SPACE=$SUPPRESS_SPACE \
                  __python_argcomplete_run "$1") )
    if [[ $? != 0 ]]; then
        unset COMPREPLY
    elif [[ $SUPPRESS_SPACE == 1 ]] && [[ "${COMPREPLY-}" =~ [=/:]$ ]]; then
        compopt -o nospace
    fi
}

complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-comparetogsinetwork
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-modeldeposition
complete -o nospace -o default -o bashdefault -F _python_argcomplete getartisspencerfano
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-spencerfano
complete -o nospace -o default -o bashdefault -F _python_argcomplete listartistimesteps
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-timesteptimes
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-make1dslicefrom3dmodel
complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodel1dslicefromcone
complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelbotyanski2017
complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelfromshen2018
complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelfromlapuente
complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelscalevelocity
complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelfullymixed
complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelsolar_rprocess
complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelfromsingletrajectory
complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodelfromparticlegridmap
complete -o nospace -o default -o bashdefault -F _python_argcomplete makeartismodel
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-maketardismodelfromartis
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-maptogrid
complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartismodeldensity
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-plotdensity
complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartismodeldeposition
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-deposition
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-describeinputmodel
complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartisestimators
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-estimators
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-exportmassfractions
complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartislightcurve
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-lightcurve
complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartislinefluxes
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-linefluxes
complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartismacroatom
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-macroatom
complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartisnltepops
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-nltepops
complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartisnonthermal
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-nonthermal
complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartisradfield
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-radfield
complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartisspectrum
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-spectrum
complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartistransitions
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-transitions
complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartisinitialcomposition
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-initialcomposition
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-writecodecomparisondata
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-setup_completions
complete -o nospace -o default -o bashdefault -F _python_argcomplete artistools-viewingangles
complete -o nospace -o default -o bashdefault -F _python_argcomplete plotartisviewingangles
