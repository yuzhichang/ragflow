#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os
import sys
import logging
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from flask import Blueprint, Flask
from werkzeug.wrappers.request import Request
from flask_cors import CORS
from itsdangerous.url_safe import URLSafeTimedSerializer as Serializer

from api.db import StatusEnum
from api.db.db_models import close_connection
from api.db.services import UserService
from api.utils import CustomJSONEncoder, commands

from flask_session import Session
from flask_login import LoginManager
from api import settings
from api.utils.api_utils import server_error_response


CORS(app, supports_credentials=True, max_age=2592000)
app.url_map.strict_slashes = False
app.json_encoder = CustomJSONEncoder
app.errorhandler(Exception)(server_error_response)

## convince for dev and debug
# app.config["LOGIN_DISABLED"] = True
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_CONTENT_LENGTH", 1024 * 1024 * 1024)
)

Session(app)
login_manager = LoginManager()
login_manager.init_app(app)

commands.register_commands(app)




@login_manager.request_loader
def load_user(web_request):
    jwt = Serializer(secret_key=settings.SECRET_KEY)
    authorization = web_request.headers.get("Authorization")
    if authorization:
        try:
            access_token = str(jwt.loads(authorization))
            user = UserService.query(
                access_token=access_token, status=StatusEnum.VALID.value
            )
            if user:
                return user[0]
            else:
                return None
        except Exception as e:
            logging.warning(f"load_user got exception {e}")
            return None
    else:
        return None


@app.teardown_request
def _db_close(exc):
    close_connection()
