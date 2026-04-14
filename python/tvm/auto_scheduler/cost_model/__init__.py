# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-import, redefined-builtin
""" Cost model that estimates the performance of programs """

from .cost_model import RandomModel, RandomModelInternal
from .xgb_model import XGBModel

import os

# Optional cost models may require heavy extra dependencies (e.g. torch/lightgbm).
# Keep base auto_scheduler import usable by default; opt in via env var when needed.
MLPModel = None
LGBModel = None
TabNetModel = None

if os.environ.get("TVM_ENABLE_OPTIONAL_COST_MODELS", "0") == "1":
    try:
        from .mlp_model import MLPModel
    except Exception:  # pylint: disable=broad-except
        MLPModel = None

    try:
        from .lgbm_model import LGBModel
    except Exception:  # pylint: disable=broad-except
        LGBModel = None

    try:
        from .tabnet_model import TabNetModel
    except Exception:  # pylint: disable=broad-except
        TabNetModel = None
