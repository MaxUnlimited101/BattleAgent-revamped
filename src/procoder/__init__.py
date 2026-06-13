################################################################################
# Define basic utilities
################################################################################

from .functional import (
    AddIndentProxy as AddIndentProxy,
    Module as Module,
    Single as Single,
    SilenceProxy as SilenceProxy,
    T as T,
    TS as TS,
    add_brackets as add_brackets,
    add_indent as add_indent,
    add_indent2 as add_indent2,
    add_indent4 as add_indent4,
    add_indent_tab as add_indent_tab,
    add_refnames as add_refnames,
    as_module as as_module,
    as_prompt as as_prompt,
    check_duplicate_keys as check_duplicate_keys,
    collect_refnames as collect_refnames,
    find_submodule as find_submodule,
    format_multiple_prompts as format_multiple_prompts,
    format_prompt as format_prompt,
    format_refnames as format_refnames,
    indent2 as indent2,
    indent4 as indent4,
    indent_tab as indent_tab,
    process_module_refname as process_module_refname,
    remove_direct_submodule as remove_direct_submodule,
    remove_submodule as remove_submodule,
    removed_submodule as removed_submodule,
    removed_submodules as removed_submodules,
    replace_prompt as replace_prompt,
    replace_submodule as replace_submodule,
    replaced_submodule as replaced_submodule,
    silence as silence,
)

__version__ = "0.0.1"
