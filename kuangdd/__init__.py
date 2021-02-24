#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2021/2/24
"""
__init__.py
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import os
import re
import json
import shutil
import collections as clt
import functools
import multiprocessing as mp
import traceback
import tempfile

if __name__ == "__main__":
    logger.info(__file__)
