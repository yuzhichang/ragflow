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

# from beartype import BeartypeConf
# from beartype.claw import beartype_all  # <-- you didn't sign up for this
# beartype_all(conf=BeartypeConf(violation_type=UserWarning))    # <-- emit warnings from all code

import faulthandler
import logging
import os
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading
import importlib
import uvicorn
from fastapi import FastAPI

from api import settings
from api.apps import app
from api.db.services.document_service import DocumentService
from api import utils

from api.db.db_models import init_database_tables as init_web_db
from api.db.init_data import init_web_data
from api.versions import get_ragflow_version
from api.utils import show_configs
from rag.settings import print_rag_settings
from rag.utils.redis_conn import RedisDistributedLock
from api.utils.log_utils import initRootLogger

stop_event = threading.Event()

def update_progress():
    redis_lock = RedisDistributedLock("update_progress", timeout=60)
    while not stop_event.is_set():
        try:
            if not redis_lock.acquire():
                continue
            DocumentService.update_progress()
            stop_event.wait(6)
        except Exception:
            logging.exception("update_progress exception")
        finally:
            redis_lock.release()

def signal_handler(sig, frame):
    logging.info("Received interrupt signal, shutting down...")
    stop_event.set()
    time.sleep(1)
    sys.exit(0)

def init_app_routes(app: FastAPI, routes_dirs: list[str]):
    for routes_dir in routes_dirs:
        for filename in os.listdir(routes_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]
                module = importlib.import_module(f"{routes_dir}.{module_name}")
                if hasattr(module, "router"):
                    app.include_router(module.router)

def get_cpu_limit():
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            cpu_quota = int(f.read().strip())

        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            cpu_period = int(f.read().strip())

        if cpu_quota > 0 and cpu_period > 0:
            return max(1, int(cpu_quota / cpu_period))
        else:
            import os
            return os.cpu_count()
    except FileNotFoundError:
        import os
        return os.cpu_count()

if __name__ == '__main__':
    faulthandler.enable()
    initRootLogger("ragflow_server")
    logging.info(r"""
        ____   ___    ______ ______ __               
       / __ \ /   |  / ____// ____// /____  _      __
      / /_/ // /| | / / __ / /_   / // __ \| | /| / /
     / _, _// ___ |/ /_/ // __/  / // /_/ /| |/ |/ / 
    /_/ |_|/_/  |_|\____//_/    /_/ \____/ |__/|__/                             

    """)
    logging.info(
        f'RAGFlow version: {get_ragflow_version()}'
    )
    logging.info(
        f'project base: {utils.file_utils.get_project_base_directory()}'
    )
    show_configs()
    settings.init_settings()
    print_rag_settings()

    # init db
    init_web_db()
    init_web_data()
    # init runtime config
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", default=False, help="RAGFlow version", action="store_true"
    )
    parser.add_argument(
        "--debug", default=False, help="debug mode", action="store_true"
    )
    args = parser.parse_args()
    if args.version:
        print(get_ragflow_version())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    thread = ThreadPoolExecutor(max_workers=1)
    thread.submit(update_progress)

    # start http server
    try:
        logging.info("RAGFlow HTTP server start...")
        app = FastAPI()
        init_app_routes(app, ["apps", "apps/sdk"])
        workers = max(get_cpu_limit() // 4, 1)
        uvicorn.run(app, host=settings.HOST_IP, port=settings.HOST_PORT, workers=workers)
    except Exception:
        traceback.print_exc()
        stop_event.set()
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGKILL)
