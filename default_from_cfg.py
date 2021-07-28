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
    #{{{ 
        return eval('cfg.%s'%self.name)
    #}}}
