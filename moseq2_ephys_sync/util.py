from os.path import join, exists
import os
from glob import glob
import pdb
import numpy as np
from warnings import warn
from moseq2_ephys_sync.workflows import get_valid_source_abbrevs


class GreaterThanOneMatchingFileError(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

class NoMatchingFilesError(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)


def verify_sources(first_source, second_source):
    if exists(first_source):
        first_source_name = os.path.splitext(os.path.basename(first_source))[0]
    else:
        first_source_name = first_source
        if first_source_name not in get_valid_source_abbrevs():
            raise ValueError(f'First source keyword {first_source_name} not recognized')
    if exists(second_source):
        second_source_name = os.path.splitext(os.path.basename(second_source))[0]
    else:
        second_source_name = second_source
        if second_source_name not in get_valid_source_abbrevs():
            raise ValueError(f'Second source keyword {second_source_name} not recognized')
    return first_source_name, second_source_name


def find_file_through_glob_and_symlink(path, pattern, exclude_patterns=None):
    """Returns path to file found that matches pattern in path, or tries to follow symlink to raw data. Must only be one that matches!
    path: path to folder with data
    pattern: glob pattern, eg *.txt for arduino data
    """

    # Check path exists
    assert os.path.exists(path), f'Path {path} does not exist'

    # Simply look for file
    files = glob(os.path.join(path, pattern))

    # Remove known confounders (eg we use '*.txt' for arduino files, but sometimes we also have 'depth_ts.txt')
    if exclude_patterns is None:
        exclude_patterns = []
    files = [f for f in files if not any([pattern in f for pattern in exclude_patterns])]

    # If no files found, maybe we have to follow a sym-linked depth-video back to the raw data directory
    if len(files) == 0 and ('mkv' in pattern or 'avi' in pattern):
        try_avi = glob(os.path.join(path, f'*depth.avi'))[0]
        if len(try_avi) == 0:
            try_mkv = glob(os.path.join(path, f'*depth.mkv'))[0]
            if len(try_mkv) == 0:
                raise RuntimeError(f'Could not find symlinked depth file in {path}')
            depth_path = try_mkv
        else:
            depth_path = try_avi

        # Follow the symlink to find desired file
        sym_path = os.readlink(depth_path)
        containing_dir = os.path.dirname(sym_path)
        files = glob(os.path.join(containing_dir, pattern))
        files = [f for f in files if not any([pattern in f for pattern in exclude_patterns])]

    # Sanitize output
    if len(files) == 0: 
        raise NoMatchingFilesError(f'Found no files in {path} matching pattern {pattern}')
    elif len(files) > 1:
        raise GreaterThanOneMatchingFileError(f'Found multiple files in {path} matching pattern {pattern}')

    return files[0]
