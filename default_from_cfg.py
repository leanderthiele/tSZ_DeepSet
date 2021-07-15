import cfg


class DefaultFromCfg :
    """
    Since default arguments are evaluated when parsing, and many of them
    are taken from cfg.py, we want to ensure that if something in cfg.py
    is changed by InitProc this is reflected whenever a default argument
    is used.
    """

    def __init__(self, name) :
        """
        name ... anything such that cfg.name is a valid identifier
        """
    #{{{
        self.name = name
    #}}}


    def __call__(self) :
        """
        Can be passed to eval/exec
        """
    #{{{ 
        return 'cfg.%s'%name
    #}}}


def SetDefaults(scope) :
    """
    To be called at the beginning of functions that could be passed DefaultFromCfg instances.
    scope ... a dict returned by locals or vars

    Returns a string that needs to be passed to exec.
    """
    #{{{
    out = ''
    for k, v in scope.items() :
        if isinstance(v, DefaultFromCfg) :
            out += '%s = %s\n'%(k, v())
    return out
    #}}}
